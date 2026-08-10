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


def resolve_normal_sign(membrane_normal: np.ndarray, patch_coords: np.ndarray, particle_position: np.ndarray) -> np.ndarray:
    """Orients the sign-ambiguous SVD normal using the particle's position relative to its local patch."""

    #weight patch points by inverse distance to the particle so nearby points (which best represent
    #the membrane right under the particle) dominate the centroid, and distant patch points - which on
    #a curved/asymmetric patch can otherwise drag the centroid off to one side - barely count
    distances = np.linalg.norm(patch_coords - particle_position, axis=1)
    weights = 1.0 / (distances + 1e-6)
    patch_centroid = np.average(patch_coords, axis=0, weights=weights)

    #vector from the (now distance-weighted) patch centroid to the particle's own coordinate - this
    #points "outward" from the membrane toward wherever the particle actually is
    centroid_to_particle = particle_position - patch_centroid

    #flip the normal if it points away from the particle rather than toward it
    if np.dot(membrane_normal, centroid_to_particle) < 0:
        membrane_normal = -membrane_normal

    return membrane_normal


def build_tangent_frame(membrane_normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Builds an arbitrary orthonormal (u, v) basis perpendicular to the given normal."""
    reference_axis = np.array([1.0, 0.0, 0.0]) if abs(membrane_normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_axis = np.cross(membrane_normal, reference_axis)
    u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(membrane_normal, u_axis)
    return u_axis, v_axis


def calculate_local_curvature(patch_coords: np.ndarray, anchor: np.ndarray, membrane_normal: np.ndarray) -> float:
    """Fits a local quadratic surface to the patch to get the sign and rough size of its curvature.

    membrane_normal must already point from the membrane toward the particle (i.e. resolve_normal_sign
    has already been applied). A negative result means the membrane bulges toward the particle - convex,
    particle on the outside, as expected. A positive result means the membrane wraps around the particle -
    concave, particle on the inside - which shouldn't happen for a real glycoprotein pick and is worth
    flagging. Validated on synthetic convex/concave test patches: a convex cap with the particle outside
    gives a negative value, the identical cap with the particle on the concave side gives the exact
    positive of that value, and a flat patch gives ~0.
    """
    u_axis, v_axis = build_tangent_frame(membrane_normal)

    #express each patch point relative to the anchor in (u, v, height-along-normal) coordinates
    relative_coords = patch_coords - anchor
    u_coords = relative_coords @ u_axis
    v_coords = relative_coords @ v_axis
    height = relative_coords @ membrane_normal

    #least-squares fit of height ~ D*u + E*v + A*u^2 + B*v^2 + C*u*v
    design_matrix = np.column_stack([u_coords, v_coords, u_coords**2, v_coords**2, u_coords * v_coords])
    coefficients, *_ = np.linalg.lstsq(design_matrix, height, rcond=None)
    _, _, quad_u, quad_v, _ = coefficients

    #A+B is the trace of the quadratic form, which doesn't depend on how (u, v) happened to be oriented
    return quad_u + quad_v


def calculate_angular_gap(patch_coords: np.ndarray, anchor: np.ndarray, membrane_normal: np.ndarray) -> float:
    """Finds the widest empty angular sector around the anchor, viewed down the membrane normal.

    A large gap means the patch has no coverage on one side - e.g. the particle sits near the edge of a
    segmented membrane fragment, or in a missing-wedge dropout - which can bias the fitted normal. Checked
    this against real data: particles with a consistently large gap across multiple patch radii turned out
    to sit right at genuine fragment edges (confirmed by looking at a much wider context radius), while
    particles with a small gap at the radius actually used for the analysis were reliable even when a
    smaller radius alone made them look borderline.
    """
    u_axis, v_axis = build_tangent_frame(membrane_normal)

    #angular position of each patch point around the anchor, in the tangent plane
    relative_coords = patch_coords - anchor
    angles = np.arctan2(relative_coords @ v_axis, relative_coords @ u_axis)

    #sort the angles and find the single largest empty wedge between consecutive points (wrapping around)
    sorted_angles = np.sort(angles)
    gaps = np.diff(sorted_angles, append=sorted_angles[0] + 2 * np.pi)
    return np.rad2deg(np.max(gaps))


def calculate_tilt_angle(particle_vector: np.ndarray, membrane_normal: np.ndarray) -> float:
    """Calculates the angle between the particle's orientation vector and the (already sign-resolved) membrane normal."""

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
    patch_radius: float,
    distance_to_membrane_threshold: float,
    angular_gap_threshold: float
) -> Tuple[List[float], List[float], List[float]]:
    """Runs the maths for all the particles."""
    #maps the segmentation mask into a a highly optimised search index so it can find the nearest membrane a lot quicker.
    tree = cKDTree(membrane_coords)

    #finds the single nearest membrane voxel for each particle, and how far away it is
    nearest_membrane_dist, indices = tree.query(particle_coords)
    closest_membrane_points = membrane_coords[indices]

    #extract a sphere around these points which will be used for finding the membrane normal
    patch_indices_list = tree.query_ball_point(closest_membrane_points, r=patch_radius)

    calculated_angles = []
    calculated_curvatures = []
    calculated_gaps = []

    #number every item in the patch indices list using enumerate
    #i catches the item number (starting from 0) and patch_indices catches the corresponding data (list of membrane voxels)
    for i, patch_indices in enumerate(patch_indices_list):
        #plausibility check: a particle further from the membrane than this threshold is a
        #distance-based outlier (mispick, wrong particle, etc) rather than a real membrane
        #protein - flag with NaN instead of computing a tilt for it
        if nearest_membrane_dist[i] > distance_to_membrane_threshold:
            calculated_angles.append(np.nan)
            calculated_curvatures.append(np.nan)
            calculated_gaps.append(np.nan)
            continue

        #here we might clean up some poor particle picks that don't have membrane within the assigned radius or ones which are near a non-segmented membrane (e.g. due to missing wedge etc).
        #instead of crashing we end up with a NaN for these, so they can be filtered out in the analysis.
        if len(patch_indices) < 3:
            calculated_angles.append(np.nan)
            calculated_curvatures.append(np.nan)
            calculated_gaps.append(np.nan)
            continue

        #put the xyz coordinates for this patch through our function to get the membrane normal
        patch_coords = membrane_coords[patch_indices]
        membrane_normal = calculate_patch_normal(patch_coords)

        #resolve which way the normal should point using this particle's own position relative to
        #its local patch - not its orientation, which is the thing we're about to measure against it
        membrane_normal = resolve_normal_sign(membrane_normal, patch_coords, particle_coords[i])

        #plausibility check: a patch with no coverage on one side (fragment edge, missing-wedge dropout)
        #can bias the fitted normal - record the gap either way, but only trust the tilt/curvature for
        #this particle if the patch actually has reasonably complete coverage around it
        angular_gap = calculate_angular_gap(patch_coords, closest_membrane_points[i], membrane_normal)
        calculated_gaps.append(angular_gap)
        if angular_gap > angular_gap_threshold:
            calculated_angles.append(np.nan)
            calculated_curvatures.append(np.nan)
            continue

        #put the eulers from the star file through our function to get the particle vector)
        rot, tilt, psi = eulers[i]
        particle_vector = euler_to_z_vector(rot, tilt, psi)

        #calculate the tilt angle between these two and add to a final list
        angle_deg = calculate_tilt_angle(particle_vector, membrane_normal)
        calculated_angles.append(angle_deg)

        #sign/size of the local curvature relative to the particle - negative (convex, particle
        #outside) is expected; positive (concave, particle inside the membrane) is worth flagging
        curvature = calculate_local_curvature(patch_coords, closest_membrane_points[i], membrane_normal)
        calculated_curvatures.append(curvature)

    return calculated_angles, calculated_curvatures, calculated_gaps


@app.command()
def analyze_tilts(
    seg_file: Annotated[Path, typer.Argument(help="Path to the segmentation .mrc file")],
    star_file: Annotated[Path, typer.Argument(help="Path to the RELION .star file")],
    seg_apx: Annotated[float, typer.Option("--seg_apx", help="Segmentation pixel size in Angstroms/pixel")],
    particles_apx: Annotated[
        float, typer.Option("--particles_apx", help="Particle coordinate pixel size in Angstroms/pixel")],
    patch_radius: Annotated[
        float, typer.Option("--patch_radius", help="Radius in Angstroms to extract local membrane patch")] = 150.0,
    distance_to_membrane_threshold: Annotated[
        float, typer.Option("--distance-to-membrane-threshold",
                             help="Maximum plausible distance in Angstroms between a particle and the nearest "
                                  "segmented membrane point (roughly the glycoprotein's ectodomain height) - "
                                  "particles further than this are flagged as distance-based outliers. Tune "
                                  "this to your own structure/dataset")] = 85.0,
    angular_gap_threshold: Annotated[
        float, typer.Option("--angular-gap-threshold",
                             help="Maximum plausible gap in degrees in the membrane coverage around a particle "
                                  "(e.g. from a segmentation fragment edge or missing-wedge dropout) - particles "
                                  "whose patch has a wider empty angular sector than this are flagged as "
                                  "coverage-based outliers. Checked at whichever --patch_radius you run with")] = 40.0,
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

    if distance_to_membrane_threshold <= 0:
        typer.echo(f"Input Error: --distance-to-membrane-threshold must be a positive number, "
                   f"got {distance_to_membrane_threshold}", err=True)
        raise typer.Exit(code=1)

    if not 0 < angular_gap_threshold <= 360:
        typer.echo(f"Input Error: --angular-gap-threshold must be a number between 0 and 360, "
                   f"got {angular_gap_threshold}", err=True)
        raise typer.Exit(code=1)

    #attempts to load the segmentation file and star and raises an error if it can't
    try:
        typer.echo(f"Loading segmentation from: {seg_file}")
        membrane_coords_vox = load_segmentation_coords(seg_file)
        typer.echo(f"Found {len(membrane_coords_vox)} membrane voxels.")

        typer.echo(f"Loading coordinates from: {star_file}")
        df, df_dict, block_name = load_star_data(star_file)
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
    #same conversion for the plausibility distance threshold
    distance_to_membrane_threshold_vox = distance_to_membrane_threshold / seg_apx

    typer.echo(f"Scaled particle coordinates (Factor: {scaling_factor:.4f}).")
    typer.echo(f"Local patch radius set to {patch_radius_vox:.2f} voxels ({patch_radius} A).")
    typer.echo(f"Distance-to-membrane plausibility threshold set to {distance_to_membrane_threshold_vox:.2f} "
               f"voxels ({distance_to_membrane_threshold} A).")
    typer.echo(f"Angular coverage gap threshold set to {angular_gap_threshold} degrees.")

    typer.echo("Building KD-Tree, extracting patches, and calculating normals...")
    calculated_angles, calculated_curvatures, calculated_gaps = compute_all_tilts(
        particle_coords=particle_coords_vox,
        eulers=eulers,
        membrane_coords=membrane_coords_vox,
        patch_radius=patch_radius_vox,
        distance_to_membrane_threshold=distance_to_membrane_threshold_vox,
        angular_gap_threshold=angular_gap_threshold
    )

    #df is the same object as the relevant entry in df_dict (or df_dict itself, if the STAR file
    #had no named blocks), so this mutation is already reflected in df_dict - nothing else to write back
    df['rlnMembraneTiltAngle'] = calculated_angles
    #negative = convex, membrane bulges toward the particle (expected); positive = concave, membrane
    #wraps around the particle (implausible pick) - exposed as its own column so it can be inspected
    #or filtered on directly (e.g. in ArtiaX) rather than only trusting it from the CLI's own diagnostics
    df['rlnMembraneCurvature'] = calculated_curvatures
    #widest empty angular sector (degrees) around the particle's patch - recorded for every particle
    #with a valid patch even if it exceeded --angular-gap-threshold and got excluded, so it's visible
    #why a given particle's tilt/curvature came back NaN, not just that it did
    df['rlnAngularCoverageGap'] = calculated_gaps

    starfile.write(df_dict, output, overwrite=True)
    typer.echo(f"Success! Analysis complete. Output saved to: {output}")

    #ArtiaX (and similar tools) can't use a NaN column as a selection criterion, so also write a
    #second STAR file with the outlier/unresolved rows dropped entirely - directly selectable as-is
    accepted_df = df[df['rlnMembraneTiltAngle'].notna()].copy()
    accepted_data = {**df_dict, block_name: accepted_df} if block_name is not None else accepted_df
    accepted_output = output.with_name(f"{output.stem}_accepted_coordinates{output.suffix}")

    starfile.write(accepted_data, accepted_output, overwrite=True)
    typer.echo(f"Accepted-only particles ({len(accepted_df)}/{len(df)}) saved to: {accepted_output}")

if __name__ == "__main__":
    app()