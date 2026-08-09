import typer
import mrcfile
import starfile
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from typing_extensions import Annotated
from typing import Tuple, List, Any

app = typer.Typer(help="GCA: Calculate tilt angles of glycoproteins relative to an irregular membrane.")


def load_segmentation_coords(seg_file: Path) -> np.ndarray:
    """Reads in the segmentation volume to numpy array."""

    #permissive=True to avoid header issues
    with mrcfile.open(seg_file, permissive=True) as mrc:
        volume = mrc.data

    #select nonzeros as most of the box is zero and we don't need those zeros
    #we need to be careful because mrcs are read zyx and we need to make this xyz otherwise tomograms will be rotated compared to coordinates
    z_idx, y_idx, x_idx = np.nonzero(volume)
    membrane_voxels = np.column_stack((x_idx, y_idx, z_idx))

    if len(membrane_voxels) == 0:
        raise ValueError("No non-zero voxels found in the segmentation mask.")

    return membrane_voxels


def load_star_data(star_file: Path) -> Tuple[pd.DataFrame, Any, str]:
    """Loads a STAR file and returns the particles dataframe"""
    #if there are multiple data blocks it will get read into a dictionary
    df_dict = starfile.read(star_file)
    is_dict = isinstance(df_dict, dict)

    # Check if it's a dictionary. If so, extract the 'particles' block (or default to the first available block).
    # If it's not a dictionary, it will have just been read in as a single table.
    block_name = 'particles' if is_dict and 'particles' in df_dict else list(df_dict.keys())[0] if is_dict else None
    df = df_dict[block_name] if block_name else df_dict

    # Check we have found our coordinates and euler angles
    required_cols = ['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ', 'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing expected column in STAR file: {col}")

    return df, df_dict, block_name

def euler_to_z_vector(rot: float, tilt: float, psi: float) -> np.ndarray:
    """Converts RELION Euler angles to a 3D vector."""

    #convert to radians for numpy
    rot_rad = np.deg2rad(rot)
    tilt_rad = np.deg2rad(tilt)

    #get cartesian coordinates (each describing how far the vector extends along each axis)
    v_x = -np.sin(rot_rad) * np.sin(tilt_rad)
    v_y = np.cos(rot_rad) * np.sin(tilt_rad)
    v_z = np.cos(tilt_rad)
    vector = np.array([v_x, v_y, v_z])
    #normalise to ensure vector has a length of exactly 1.
    return vector / np.linalg.norm(vector)


def calculate_patch_normal(patch_coords: np.ndarray) -> np.ndarray:
    """Calculates the normal vector of a 3D point cloud using PCA (SVD)."""
    # find the center of patch coordinates
    centered_patch = patch_coords - np.mean(patch_coords, axis=0)

    # Perform SVD to find the vector of least variance
    #Vt describes the following :
    #row0 = longest axis of patch (maxiumum spread)
    #row1 = width of patch (secondary spread)
    #row2 = thickness of patch (minimum spread)
    _, _, Vt = np.linalg.svd(centered_patch)
    #take the last item of this array (e.g. the minimum spread vector pointing through the membrane e.g. the theoretical 0 tilt for our glycoprotein)
    normal = Vt[-1, :]
    #normalise as before to make sure the vector length is 1
    return normal / np.linalg.norm(normal)


def calculate_tilt_angle(particle_vector: np.ndarray, membrane_normal: np.ndarray) -> float:
    """Calculates the angle between the particle and the outward-facing membrane normal."""

    # Making sure membrane vector orientation isn't flipped relative to glycoprotein
    # If the dot product is negative, then the vectors point in opposite directions
    if np.dot(particle_vector, membrane_normal) < 0:
        #if so then flip it
        membrane_normal = -membrane_normal

    #For two unit vectors, the dot product is exactly equal to the cosine of the angle between them:
    #our vectors were already normalised earlier which is why we can do this
    dot_product = np.dot(particle_vector, membrane_normal)

    #Clipping prevents floating-point rounding errors
    clipped_dot = np.clip(dot_product, -1.0, 1.0)

    #converting to radians then degrees
    angle_rad = np.arccos(clipped_dot)
    angle_deg = np.rad2deg(angle_rad)

    return angle_deg

def compute_all_tilts(
    particle_coords: np.ndarray,
    eulers: np.ndarray,
    membrane_coords: np.ndarray,
    patch_radius: float
) -> List[float]:
    """Runs the maths for all the particles."""
    #maps the segmentation mask into a a highly optimised search index so it can find the nearest membrane a lot quicker.
    tree = cKDTree(membrane_coords)

    #finds the single nearest membrane voxel for each particle
    _, indices = tree.query(particle_coords)
    closest_membrane_points = membrane_coords[indices]

    #extract a sphere around these points which will be used for finding the membrane normal
    patch_indices_list = tree.query_ball_point(closest_membrane_points, r=patch_radius)

    calculated_angles = []

    #number every item in the patch indices list using enumerate
    #i catches the item number (starting from 0) and patch_indices catches the corresponding data (list of membrane voxels)
    for i, patch_indices in enumerate(patch_indices_list):
        #here we might clean up some poor particle picks that don't have membrane within the assigned radius or ones which are near a non-segmented membrane (e.g. due to missing wedge etc).
        #instead of crashing we end up with a NaN for these, so they can be filtered out in the analysis.
        if len(patch_indices) < 3:
            calculated_angles.append(np.nan)
            continue

        #put the xyz coordinates for this patch through our function to get the membrane normal
        patch_coords = membrane_coords[patch_indices]
        membrane_normal = calculate_patch_normal(patch_coords)

        #put the eulers from the star file through our function to get the particle vector)
        rot, tilt, psi = eulers[i]
        particle_vector = euler_to_z_vector(rot, tilt, psi)

        #calculate the tilt angle between these two and add to a final list
        angle_deg = calculate_tilt_angle(particle_vector, membrane_normal)
        calculated_angles.append(angle_deg)

    return calculated_angles


@app.command()
def analyze_tilts(
    seg_file: Annotated[Path, typer.Argument(help="Path to the segmentation .mrc file")],
    star_file: Annotated[Path, typer.Argument(help="Path to the RELION .star file")],
    seg_apx: Annotated[float, typer.Option("--seg_apx", help="Segmentation pixel size in Angstroms/pixel")],
    particles_apx: Annotated[
        float, typer.Option("--particles_apx", help="Particle coordinate pixel size in Angstroms/pixel")],
    patch_radius: Annotated[
        float, typer.Option("--patch_radius", help="Radius in Angstroms to extract local membrane patch")] = 150.0,
    output: Annotated[Path, typer.Option("--output", "-o", help="Output STAR file to save results")] = Path(
        "tilts_output.star")
):
    """
    Load data, map spaces, run PCA on local membrane patches, and calculate particle tilt angles.
    """

    #check inputs exist and are sane before attempting to load/process anything
    if not seg_file.exists():
        typer.echo(f"Input Error: Segmentation file not found: {seg_file}", err=True)
        raise typer.Exit(code=1)

    if not star_file.exists():
        typer.echo(f"Input Error: STAR file not found: {star_file}", err=True)
        raise typer.Exit(code=1)

    if seg_apx <= 0:
        typer.echo(f"Input Error: --seg_apx must be a positive number, got {seg_apx}", err=True)
        raise typer.Exit(code=1)

    if particles_apx <= 0:
        typer.echo(f"Input Error: --particles_apx must be a positive number, got {particles_apx}", err=True)
        raise typer.Exit(code=1)

    if patch_radius <= 0:
        typer.echo(f"Input Error: --patch_radius must be a positive number, got {patch_radius}", err=True)
        raise typer.Exit(code=1)

    #attempts to load the segmentation file and star and raises an error if it can't
    try:
        typer.echo(f"Loading segmentation from: {seg_file}")
        membrane_coords_vox = load_segmentation_coords(seg_file)
        typer.echo(f"Found {len(membrane_coords_vox)} membrane voxels.")

        typer.echo(f"Loading coordinates from: {star_file}")
        df, df_dict, _ = load_star_data(star_file)
        typer.echo(f"Loaded {len(df)} particles.")

    except (ValueError, KeyError) as e:
        typer.echo(f"Data Error: {e}", err=True)
        raise typer.Exit(code=1)

    #calculate scale factor to relate particles to the segmentation
    scaling_factor = particles_apx / seg_apx
    #convert the star coordinates accordingly
    particle_coords_vox = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy() * scaling_factor
    eulers = df[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].to_numpy()
    #figure out how many voxels the membrane extraction from our seg volume needs to be
    patch_radius_vox = patch_radius / seg_apx

    typer.echo(f"Scaled particle coordinates (Factor: {scaling_factor:.4f}).")
    typer.echo(f"Local patch radius set to {patch_radius_vox:.2f} voxels ({patch_radius} A).")

    typer.echo("Building KD-Tree, extracting patches, and calculating normals...")
    calculated_angles = compute_all_tilts(
        particle_coords=particle_coords_vox,
        eulers=eulers,
        membrane_coords=membrane_coords_vox,
        patch_radius=patch_radius_vox
    )

    #df is the same object as the relevant entry in df_dict (or df_dict itself, if the STAR file
    #had no named blocks), so this mutation is already reflected in df_dict - nothing else to write back
    df['rlnMembraneTiltAngle'] = calculated_angles

    starfile.write(df_dict, output, overwrite=True)
    typer.echo(f"Success! Analysis complete. Output saved to: {output}")

if __name__ == "__main__":
    app()