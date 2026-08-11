import functools
import typer
import mrcfile
import starfile
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from scipy.spatial import cKDTree
from typing_extensions import Annotated
from typing import Dict, Tuple, List, Any, Optional

app = typer.Typer(help="GTA: Calculate tilt angles of glycoproteins relative to an irregular membrane.")

#internal tuning for adaptive patch radius search - validated against real data rather than exposed
#as CLI options, to keep the user-facing interface focused on just min/max radius and on/off
PATCH_RADIUS_SEARCH_STEP_A = 5.0
STABILITY_TOLERANCE_DEG = 8.0
STABILITY_WINDOW_A = 40.0


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


def find_stable_patch_radius(
    candidate_points: np.ndarray,
    candidate_dists: np.ndarray,
    anchor: np.ndarray,
    particle_position: np.ndarray,
    particle_vector: np.ndarray,
    min_radius: float,
    max_radius: float,
    radius_step: float,
    angular_gap_threshold: float
) -> Optional[dict]:
    """Searches for the smallest patch radius between min_radius and max_radius at which the tilt
    angle has actually settled, instead of just the first radius that has enough points and an
    acceptable coverage gap.

    That distinction matters: a patch can satisfy a point-count and gap check while its fitted
    normal is still actively drifting as the patch grows (checked on real data - a particle sitting
    on an unusually tightly-curved patch passed both checks at one radius but its tilt angle then
    swung by over 60 degrees before settling much further out). So here, a candidate radius is only
    accepted once the tilt angle stays within STABILITY_TOLERANCE_DEG of its own reading for every
    other candidate radius up to STABILITY_WINDOW_A further out - i.e. growing the patch further
    stops changing the answer.

    Falls back to the largest radius that at least has an acceptable gap if nothing stabilises.
    Returns None only if not even max_radius gives a usable (>=3 point, gap<=threshold) patch.
    """
    radii = np.append(np.arange(min_radius, max_radius, radius_step), max_radius)

    #evaluate every candidate radius once - every one with a fittable (>=3 point) patch, regardless
    #of whether its gap passes, so there's still something to report if none of them do
    evaluations = []
    for radius in radii:
        patch = candidate_points[candidate_dists <= radius]
        if len(patch) < 3:
            continue
        raw_normal = calculate_patch_normal(patch)
        normal = resolve_normal_sign(raw_normal, patch, particle_position)
        gap = calculate_angular_gap(patch, anchor, normal)
        angle = calculate_tilt_angle(particle_vector, normal)
        evaluations.append(dict(radius=radius, patch=patch, normal=normal, gap=gap, angle=angle))

    if not evaluations:
        return None

    passing = [e for e in evaluations if e['gap'] <= angular_gap_threshold]

    #smallest radius whose angle stays flat across every later candidate within the lookahead window
    for candidate in passing:
        lookahead = [e for e in passing if candidate['radius'] < e['radius'] <= candidate['radius'] + STABILITY_WINDOW_A]
        if not lookahead:
            continue
        if all(abs(e['angle'] - candidate['angle']) <= STABILITY_TOLERANCE_DEG for e in lookahead):
            return dict(radius=candidate['radius'], patch=candidate['patch'], normal=candidate['normal'],
                        gap=candidate['gap'], valid=True)

    #nothing confirmed stable - fall back to the largest radius that at least passed the gap check
    if passing:
        best = passing[-1]
        return dict(radius=best['radius'], patch=best['patch'], normal=best['normal'], gap=best['gap'], valid=True)

    #nothing passed the gap check even at max_radius - still report the gap there for transparency
    worst = evaluations[-1]
    return dict(radius=worst['radius'], patch=None, normal=None, gap=worst['gap'], valid=False)


def compute_all_tilts(
    particle_coords: np.ndarray,
    eulers: np.ndarray,
    membrane_coords: np.ndarray,
    distance_to_membrane_threshold: float,
    angular_gap_threshold: float,
    max_tilt_angle: float,
    adaptive_patch_radius: bool,
    patch_radius: float,
    min_patch_radius: float,
    max_patch_radius: float,
    patch_radius_step: float
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Runs the maths for all the particles.

    If adaptive_patch_radius is True, patch_radius is ignored and each particle instead searches
    between min_patch_radius and max_patch_radius for a patch size at which its tilt angle has
    settled (see find_stable_patch_radius). Otherwise every particle uses the single fixed
    patch_radius, as before.
    """
    #maps the segmentation mask into a a highly optimised search index so it can find the nearest membrane a lot quicker.
    tree = cKDTree(membrane_coords)

    #finds the single nearest membrane voxel for each particle, and how far away it is
    nearest_membrane_dist, indices = tree.query(particle_coords)
    closest_membrane_points = membrane_coords[indices]

    #query once at the largest radius any particle could need - the search ceiling if adaptive,
    #or the single fixed radius otherwise - then narrow down per-particle from there rather than
    #re-querying the tree for every candidate radius in the adaptive search
    query_radius = max_patch_radius if adaptive_patch_radius else patch_radius
    patch_indices_list = tree.query_ball_point(closest_membrane_points, r=query_radius)

    calculated_angles = []
    calculated_curvatures = []
    calculated_gaps = []
    calculated_radii = []

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
            calculated_radii.append(np.nan)
            continue

        candidate_points = membrane_coords[patch_indices]

        #put the eulers from the star file through our function to get the particle vector) - needed
        #up front now since the adaptive search itself tracks the tilt angle to test for stability
        rot, tilt, psi = eulers[i]
        particle_vector = euler_to_z_vector(rot, tilt, psi)

        if adaptive_patch_radius:
            candidate_dists = np.linalg.norm(candidate_points - closest_membrane_points[i], axis=1)
            result = find_stable_patch_radius(
                candidate_points, candidate_dists, closest_membrane_points[i], particle_coords[i],
                particle_vector, min_patch_radius, max_patch_radius, patch_radius_step, angular_gap_threshold
            )
            if result is None:
                calculated_angles.append(np.nan)
                calculated_curvatures.append(np.nan)
                calculated_gaps.append(np.nan)
                calculated_radii.append(np.nan)
                continue
        else:
            #here we might clean up some poor particle picks that don't have membrane within the assigned radius or ones which are near a non-segmented membrane (e.g. due to missing wedge etc).
            #instead of crashing we end up with a NaN for these, so they can be filtered out in the analysis.
            if len(candidate_points) < 3:
                calculated_angles.append(np.nan)
                calculated_curvatures.append(np.nan)
                calculated_gaps.append(np.nan)
                calculated_radii.append(np.nan)
                continue

            #put the xyz coordinates for this patch through our function to get the membrane normal
            raw_normal = calculate_patch_normal(candidate_points)

            #resolve which way the normal should point using this particle's own position relative to
            #its local patch - not its orientation, which is the thing we're about to measure against it
            normal = resolve_normal_sign(raw_normal, candidate_points, particle_coords[i])
            gap = calculate_angular_gap(candidate_points, closest_membrane_points[i], normal)
            result = dict(radius=patch_radius, patch=candidate_points, normal=normal, gap=gap,
                          valid=gap <= angular_gap_threshold)

        #plausibility check: a patch with no coverage on one side (fragment edge, missing-wedge dropout)
        #can bias the fitted normal - record the gap and radius used either way, but only trust the
        #tilt/curvature for this particle if the patch actually has reasonably complete coverage
        calculated_gaps.append(result['gap'])
        calculated_radii.append(result['radius'])
        if not result['valid']:
            calculated_angles.append(np.nan)
            calculated_curvatures.append(np.nan)
            continue

        patch_coords = result['patch']
        membrane_normal = result['normal']

        #calculate the tilt angle between these two and add to a final list
        angle_deg = calculate_tilt_angle(particle_vector, membrane_normal)

        #plausibility check: a real membrane-anchored glycoprotein can't tilt anywhere near 90
        #degrees without its ectodomain clashing into the membrane - particles this tilted are
        #typically junk picks (e.g. sitting on segmented ice contamination rather than membrane)
        #rather than real, if unusual, biology. This is a different kind of check to the others
        #above: it's about whether the *result* looks biologically plausible, not whether the
        #patch itself was good enough to trust - so the membrane geometry (curvature) is still
        #reported below even when the angle itself gets excluded
        calculated_angles.append(angle_deg if angle_deg <= max_tilt_angle else np.nan)

        #sign/size of the local curvature relative to the particle - negative (convex, particle
        #outside) is expected; positive (concave, particle inside the membrane) is worth flagging
        curvature = calculate_local_curvature(patch_coords, closest_membrane_points[i], membrane_normal)
        calculated_curvatures.append(curvature)

    return calculated_angles, calculated_curvatures, calculated_gaps, calculated_radii


def discover_tomogram_sets(data_dir: Path, require_tomograms: bool) -> List[dict]:
    """Finds segmentation/starfile pairs (or, if require_tomograms, tomogram/segmentation/starfile
    triples) that share an exact filename stem, across the segmentations/, starfiles/, and (if
    required) tomograms/ subdirectories of data_dir - each is one tomogram to analyze. Tomograms
    are only needed for --prepare-review's image rendering; plain tilt-angle analysis only needs
    the segmentation and particle coordinates."""
    seg_dir = data_dir / "segmentations"
    star_dir = data_dir / "starfiles"
    for d in (seg_dir, star_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Expected subdirectory not found: {d}")

    seg_files = {p.stem: p for p in seg_dir.glob("*.mrc")}
    star_files = {p.stem: p for p in star_dir.glob("*.star")}

    if require_tomograms:
        tomo_dir = data_dir / "tomograms"
        if not tomo_dir.is_dir():
            raise FileNotFoundError(f"Expected subdirectory not found: {tomo_dir} (required for --prepare-review)")
        tomo_files = {p.stem: p for p in tomo_dir.glob("*.mrc")}
        common_stems = sorted(set(tomo_files) & set(seg_files) & set(star_files))
        all_stems = set(tomo_files) | set(seg_files) | set(star_files)
    else:
        tomo_files = {}
        common_stems = sorted(set(seg_files) & set(star_files))
        all_stems = set(seg_files) | set(star_files)

    skipped = sorted(all_stems - set(common_stems))
    if skipped:
        typer.echo(f"Note: skipping {len(skipped)} file(s) without a matching stem in all required "
                   f"subdirectories: {skipped}")
    if not common_stems:
        kind = "tomogram/segmentation/starfile triples" if require_tomograms else "segmentation/starfile pairs"
        raise ValueError(f"No matching {kind} found under {data_dir}")

    result = []
    for s in common_stems:
        entry = dict(stem=s, segmentation=seg_files[s], starfile=star_files[s])
        if require_tomograms:
            entry['tomogram'] = tomo_files[s]
        result.append(entry)
    return result


def _process_tomogram_for_analysis(
    tomo_set: dict, output_dir: Path, seg_apx: float, particles_apx: float,
    distance_to_membrane_threshold: float, angular_gap_threshold: float, max_tilt_angle: float,
    adaptive_patch_radius: bool, patch_radius: float, min_patch_radius: float, max_patch_radius: float,
    prepare_review: bool, inputs_dir: Optional[Path],
) -> dict:
    """Runs the accept/reject tilt analysis for one tomogram and writes its two output STAR files
    to output_dir - deliberately not into the starfiles/ input directory itself, since that would
    get re-scanned as spurious input on the next run. If prepare_review is set, also renders the
    tomogram density images `gta review` needs and writes them to inputs_dir/<stem>/. Returns a
    small summary dict for the CLI's own progress reporting."""
    stem = tomo_set['stem']
    typer.echo(f"[{stem}] loading segmentation + star...")
    membrane_coords_vox = load_segmentation_coords(tomo_set['segmentation'])
    df, df_dict, block_name = load_star_data(tomo_set['starfile'])

    scaling_factor = particles_apx / seg_apx
    particle_coords_vox = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy() * scaling_factor
    eulers = df[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].to_numpy()

    patch_radius_vox = patch_radius / seg_apx
    min_patch_radius_vox = min_patch_radius / seg_apx
    max_patch_radius_vox = max_patch_radius / seg_apx
    patch_radius_step_vox = PATCH_RADIUS_SEARCH_STEP_A / seg_apx
    distance_to_membrane_threshold_vox = distance_to_membrane_threshold / seg_apx

    typer.echo(f"[{stem}] calculating tilt angles for {len(df)} particles...")
    calculated_angles, calculated_curvatures, calculated_gaps, calculated_radii = compute_all_tilts(
        particle_coords=particle_coords_vox,
        eulers=eulers,
        membrane_coords=membrane_coords_vox,
        distance_to_membrane_threshold=distance_to_membrane_threshold_vox,
        angular_gap_threshold=angular_gap_threshold,
        max_tilt_angle=max_tilt_angle,
        adaptive_patch_radius=adaptive_patch_radius,
        patch_radius=patch_radius_vox,
        min_patch_radius=min_patch_radius_vox,
        max_patch_radius=max_patch_radius_vox,
        patch_radius_step=patch_radius_step_vox,
    )

    df['rlnOriginalIndex'] = np.arange(len(df))
    df['rlnMembraneTiltAngle'] = calculated_angles
    df['rlnMembraneCurvature'] = calculated_curvatures
    df['rlnAngularCoverageGap'] = calculated_gaps
    df['rlnPatchRadiusUsed'] = np.array(calculated_radii) * seg_apx

    output = output_dir / f"{stem}_tilts_output.star"
    starfile.write(df_dict, output, overwrite=True)

    accepted_df = df[df['rlnMembraneTiltAngle'].notna()].copy()
    accepted_data = {**df_dict, block_name: accepted_df} if block_name is not None else accepted_df
    accepted_output = output.with_name(f"{output.stem}_accepted_coordinates{output.suffix}")
    starfile.write(accepted_data, accepted_output, overwrite=True)

    typer.echo(f"[{stem}] {len(accepted_df)}/{len(df)} particles accepted. Saved {output.name} and "
               f"{accepted_output.name}.")

    n_review_images = None
    if prepare_review:
        #lazy import: gta.review imports back from this module at load time (for `app` and the
        #geometry functions), so importing it here rather than at module level avoids relying on
        #import order between the two circularly-dependent modules
        from gta.review import render_review_data_for_tomogram
        review_records = render_review_data_for_tomogram(
            tomo_set, inputs_dir, seg_apx, particles_apx,
            distance_to_membrane_threshold, angular_gap_threshold, max_tilt_angle,
            adaptive_patch_radius, patch_radius, min_patch_radius, max_patch_radius,
        )
        n_review_images = len(review_records)

    return dict(stem=stem, n_particles=len(df), n_accepted=len(accepted_df), n_review_images=n_review_images)


@app.command()
def analyze_tilts(
    data_dir: Annotated[Path, typer.Argument(
        help="Directory containing segmentations/ and starfiles/ subdirectories (and tomograms/ too, "
             "if --prepare-review is set) - matching files (same stem) across them are treated as one "
             "tomogram to analyze")],
    seg_apx: Annotated[float, typer.Option("--seg_apx", help="Segmentation pixel size in Angstroms/pixel")],
    particles_apx: Annotated[
        float, typer.Option("--particles_apx", help="Particle coordinate pixel size in Angstroms/pixel")],
    patch_radius: Annotated[
        float, typer.Option("--patch_radius",
                             help="Fixed radius in Angstroms for the local membrane patch - only used when "
                                  "--no-adaptive-patch-radius is set")] = 150.0,
    min_patch_radius: Annotated[
        float, typer.Option("--min-patch-radius",
                             help="Smallest patch radius in Angstroms to try when searching for a stable patch "
                                  "size - only used when adaptive patch radius selection is on")] = 40.0,
    max_patch_radius: Annotated[
        float, typer.Option("--max-patch-radius",
                             help="Largest patch radius in Angstroms to try, and the ceiling the search falls "
                                  "back to if no radius stabilises - only used when adaptive patch radius "
                                  "selection is on")] = 150.0,
    adaptive_patch_radius: Annotated[
        bool, typer.Option("--adaptive-patch-radius/--no-adaptive-patch-radius",
                            help="Automatically pick the smallest patch radius per particle at which its tilt "
                                 "angle has settled, searching between --min-patch-radius and --max-patch-radius, "
                                 "instead of using a single fixed --patch_radius for every particle")] = True,
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
                                  "coverage-based outliers")] = 40.0,
    max_tilt_angle: Annotated[
        float, typer.Option("--max-tilt-angle",
                             help="Maximum biologically plausible tilt angle in degrees - a real membrane-anchored "
                                  "glycoprotein can't tilt close to 90 degrees without its ectodomain clashing "
                                  "into the membrane, so particles tilted more than this are flagged as implausible "
                                  "(typically junk picks, e.g. on segmented ice contamination). Tune this to your "
                                  "own structure/dataset")] = 80.0,
    prepare_review: Annotated[
        bool, typer.Option("--prepare-review",
                            help="Also render the tomogram density images `gta review` needs, writing "
                                 "them to data_dir/gta_review_inputs/. Requires a tomograms/ "
                                 "subdirectory under data_dir (not needed otherwise, since plain "
                                 "tilt-angle calculation only needs the segmentation and particle "
                                 "coordinates). Run `gta review data_dir` afterwards to triage the "
                                 "results by eye - it never computes anything itself.")] = False,
    workers: Annotated[
        int, typer.Option("--workers", "-j",
                           help="Number of tomograms to analyze in parallel (they're fully independent "
                                "of each other). With --prepare-review, each worker also holds one "
                                "whole tomogram's density volume in memory at once, so raise this "
                                "cautiously if RAM is limited. 1 disables parallelism and analyzes "
                                "tomograms one at a time.")] = 4,
):
    """
    Computes tilt angles for every tomogram found under data_dir - PCA on local membrane patches vs.
    each particle's own orientation - and writes two output STAR files per tomogram to
    data_dir/tilts_output/. Batches over every matched segmentation/starfile pair found under
    data_dir, in parallel (see --workers).

    Pass --prepare-review to also render the tomogram density images `gta review` needs (this
    additionally requires a tomograms/ subdirectory under data_dir) - then run `gta review data_dir`
    afterwards to triage the results by eye; it never computes anything itself.
    """
    if not data_dir.is_dir():
        typer.echo(f"Input Error: {data_dir} is not a directory", err=True)
        raise typer.Exit(code=1)
    data_dir = data_dir.resolve()

    if seg_apx <= 0:
        typer.echo(f"Input Error: --seg_apx must be a positive number, got {seg_apx}", err=True)
        raise typer.Exit(code=1)

    if particles_apx <= 0:
        typer.echo(f"Input Error: --particles_apx must be a positive number, got {particles_apx}", err=True)
        raise typer.Exit(code=1)

    if patch_radius <= 0:
        typer.echo(f"Input Error: --patch_radius must be a positive number, got {patch_radius}", err=True)
        raise typer.Exit(code=1)

    if min_patch_radius <= 0:
        typer.echo(f"Input Error: --min-patch-radius must be a positive number, got {min_patch_radius}", err=True)
        raise typer.Exit(code=1)

    if max_patch_radius <= min_patch_radius:
        typer.echo(f"Input Error: --max-patch-radius ({max_patch_radius}) must be greater than "
                   f"--min-patch-radius ({min_patch_radius})", err=True)
        raise typer.Exit(code=1)

    if distance_to_membrane_threshold <= 0:
        typer.echo(f"Input Error: --distance-to-membrane-threshold must be a positive number, "
                   f"got {distance_to_membrane_threshold}", err=True)
        raise typer.Exit(code=1)

    if not 0 < angular_gap_threshold <= 360:
        typer.echo(f"Input Error: --angular-gap-threshold must be a number between 0 and 360, "
                   f"got {angular_gap_threshold}", err=True)
        raise typer.Exit(code=1)

    if not 0 < max_tilt_angle <= 180:
        typer.echo(f"Input Error: --max-tilt-angle must be a number between 0 and 180, "
                   f"got {max_tilt_angle}", err=True)
        raise typer.Exit(code=1)

    if workers < 1:
        typer.echo("Input Error: --workers must be at least 1", err=True)
        raise typer.Exit(code=1)

    try:
        tomo_sets = discover_tomogram_sets(data_dir, require_tomograms=prepare_review)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Input Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(tomo_sets)} tomogram(s) to analyze: {[s['stem'] for s in tomo_sets]}")

    #deliberately not written into starfiles/ itself - that directory gets glob-scanned as input on
    #the next run, so writing outputs there would have them show up as (harmlessly skipped, but
    #noisy) unmatched files every time this is re-run
    output_dir = data_dir / "tilts_output"
    output_dir.mkdir(exist_ok=True)

    inputs_dir = None
    if prepare_review:
        inputs_dir = data_dir / "gta_review_inputs"
        inputs_dir.mkdir(exist_ok=True)

    process_one = functools.partial(
        _process_tomogram_for_analysis, output_dir=output_dir, seg_apx=seg_apx, particles_apx=particles_apx,
        distance_to_membrane_threshold=distance_to_membrane_threshold,
        angular_gap_threshold=angular_gap_threshold, max_tilt_angle=max_tilt_angle,
        adaptive_patch_radius=adaptive_patch_radius, patch_radius=patch_radius,
        min_patch_radius=min_patch_radius, max_patch_radius=max_patch_radius,
        prepare_review=prepare_review, inputs_dir=inputs_dir,
    )
    if workers == 1 or len(tomo_sets) == 1:
        summaries = [process_one(tomo_set) for tomo_set in tomo_sets]
    else:
        typer.echo(f"Analyzing up to {min(workers, len(tomo_sets))} tomogram(s) at a time...")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            summaries = list(executor.map(process_one, tomo_sets))

    total_particles = sum(s['n_particles'] for s in summaries)
    total_accepted = sum(s['n_accepted'] for s in summaries)
    typer.echo(f"\nDone: {total_accepted}/{total_particles} particles accepted across "
               f"{len(tomo_sets)} tomogram(s). STAR files saved to {output_dir}.")

    if prepare_review:
        total_images = sum(s['n_review_images'] for s in summaries)
        typer.echo(f"Prepared {total_images} particle(s) for review under {inputs_dir}.")
        typer.echo(f"Run `gta review {data_dir}` to triage them.")

#imported for its side effect of registering the `review` command on `app` above - placed here,
#after `app` is fully defined, to avoid a circular import at module load time
from gta import review  # noqa: E402,F401

if __name__ == "__main__":
    app()