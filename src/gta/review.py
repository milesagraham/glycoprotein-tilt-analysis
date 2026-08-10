"""
gta review: a local web app for rapidly accepting/rejecting particles by eye, using real
tomogram density slices oriented to each particle's fitted membrane normal.

Design notes:
- All the expensive work (adaptive patch-radius search, tomogram slicing, image rendering) runs
  once up front, before the server starts. The review UI itself only ever serves pre-rendered
  static images, so keying through hundreds of particles has no per-interaction lag.
- Reuses the exact same geometry functions as `gta analyze-tilts` (imported from gta.cli), so a
  particle that would be accepted/rejected by the CLI is accepted/rejected here identically -
  review is a second, manual pass on top of the same automatic criteria, not a separate pipeline.
"""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import mrcfile
import starfile
import typer
from typing_extensions import Annotated
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates

from gta.cli import (
    app,
    load_segmentation_coords,
    load_star_data,
    euler_to_z_vector,
    calculate_patch_normal,
    resolve_normal_sign,
    calculate_angular_gap,
    calculate_local_curvature,
    calculate_tilt_angle,
    find_stable_patch_radius,
    build_tangent_frame,
    PATCH_RADIUS_SEARCH_STEP_A,
)

#review image geometry - matches the 3 orthogonal, normal-referenced views from the report (Fig. 8)
BOX_A = 300.0
BOX_PIXELS = 170
ARROW_LEN_A = 70.0

#wider single-slice context view (report Fig. 10) - shows where on the virus/vesicle a pick sits,
#not just its immediate membrane patch. Sampled close to the tomogram's native pixel size rather
#than oversampled, since there's no real benefit to interpolating finer than the data at this scale.
CONTEXT_BOX_A = 2000.0
CONTEXT_BOX_PIXELS = 220
CONTEXT_ARROW_LEN_A = 300.0


def discover_tomogram_sets(data_dir: Path) -> List[dict]:
    """Finds tomogram/segmentation/starfile triples that share an exact filename stem across the
    tomograms/, segmentations/, and starfiles/ subdirectories - each is one tomogram to review."""
    tomo_dir = data_dir / "tomograms"
    seg_dir = data_dir / "segmentations"
    star_dir = data_dir / "starfiles"
    for d in (tomo_dir, seg_dir, star_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"Expected subdirectory not found: {d}")

    tomo_files = {p.stem: p for p in tomo_dir.glob("*.mrc")}
    seg_files = {p.stem: p for p in seg_dir.glob("*.mrc")}
    star_files = {p.stem: p for p in star_dir.glob("*.star")}

    common_stems = sorted(set(tomo_files) & set(seg_files) & set(star_files))
    all_stems = set(tomo_files) | set(seg_files) | set(star_files)
    skipped = sorted(all_stems - set(common_stems))
    if skipped:
        typer.echo(f"Note: skipping {len(skipped)} file(s) without a matching stem in all three "
                   f"subdirectories: {skipped}")
    if not common_stems:
        raise ValueError(f"No matching tomogram/segmentation/starfile triples found under {data_dir}")

    return [dict(stem=s, tomogram=tomo_files[s], segmentation=seg_files[s], starfile=star_files[s])
            for s in common_stems]


def _extract_slice(tomo: np.ndarray, center: np.ndarray, axis1: np.ndarray, axis2: np.ndarray,
                    grid_a: np.ndarray, grid_b: np.ndarray) -> np.ndarray:
    points = (center[None, None, :] + grid_a[..., None] * axis1[None, None, :]
              + grid_b[..., None] * axis2[None, None, :])
    coords = np.stack([points[..., 2], points[..., 1], points[..., 0]], axis=0)  # z,y,x order
    return map_coordinates(tomo, coords, order=1, mode='nearest')


def _draw_panel(ax, img: np.ndarray, particle_2d: Optional[np.ndarray], normal_2d: Optional[np.ndarray],
                 title: str) -> None:
    vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax,
              extent=[-BOX_A / 2, BOX_A / 2, -BOX_A / 2, BOX_A / 2])
    ax.scatter(0, 0, c='#f9a825', s=90, marker='+', linewidth=2.2, zorder=5)
    if normal_2d is not None:
        ax.annotate('', xy=tuple(normal_2d * ARROW_LEN_A), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color='#1f77b4', linewidth=2.6))
    if particle_2d is not None:
        ax.annotate('', xy=tuple(particle_2d * ARROW_LEN_A), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-|>', color='#d1495b', linewidth=2.6))
    ax.set_title(title, fontsize=10, color='#c7d0d1')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save_review_panel(
    path: Path,
    img_top: np.ndarray, img_true: np.ndarray, img_ortho: np.ndarray,
    pv_top: np.ndarray, pv_true: np.ndarray, pv_ortho: np.ndarray,
) -> None:
    """Renders the same 3 orthogonal views used in the diagnostic report (Fig. 8): a top-down look
    down the membrane normal, a side view aligned to the particle's own azimuth (shows the true,
    undistorted 3D tilt angle), and the side view 90 degrees around the normal from that - together
    giving full 3D context instead of one view that can make a plausible tilt look wrong."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor='#14191a')
    _draw_panel(axes[0], img_top, pv_top, None, "top-down (looking down normal)")
    _draw_panel(axes[1], img_true, pv_true, np.array([0, 1]), "side - true angle")
    _draw_panel(axes[2], img_ortho, pv_ortho, np.array([0, 1]), "side - orthogonal (90° around normal)")
    fig.tight_layout(pad=0.6)
    fig.savefig(path, dpi=105, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def _save_context_panel(path: Path, img: np.ndarray, particle_2d: np.ndarray) -> None:
    """Wider single-slice context view (report Fig. 10): same normal-referenced orientation as the
    close-up 'true angle' panel, just zoomed out enough to show the surrounding virus/vesicle."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.6), facecolor='#14191a')
    vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, origin='lower', cmap='gray', vmin=vmin, vmax=vmax,
              extent=[-CONTEXT_BOX_A / 2, CONTEXT_BOX_A / 2, -CONTEXT_BOX_A / 2, CONTEXT_BOX_A / 2])
    ax.scatter(0, 0, c='#f9a825', s=140, marker='+', linewidth=2.6, zorder=5)
    ax.annotate('', xy=tuple(particle_2d * CONTEXT_ARROW_LEN_A), xytext=(0, 0),
                arrowprops=dict(arrowstyle='-|>', color='#d1495b', linewidth=2.4))
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=105, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def analyze_tomogram_for_review(
    tomo_set: dict, cache_dir: Path, seg_apx: float, particles_apx: float,
    distance_to_membrane_threshold: float, angular_gap_threshold: float, max_tilt_angle: float,
    adaptive_patch_radius: bool, patch_radius: float, min_patch_radius: float, max_patch_radius: float,
) -> List[dict]:
    """Runs the same accept/reject pipeline as `gta analyze-tilts` for one tomogram, then renders
    a review image for every particle that passes it. Returns one record dict per accepted particle."""
    stem = tomo_set['stem']
    typer.echo(f"[{stem}] loading segmentation + star...")
    membrane_coords_vox = load_segmentation_coords(tomo_set['segmentation'])
    df, _, _ = load_star_data(tomo_set['starfile'])

    scaling_factor = particles_apx / seg_apx
    particle_coords_vox = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy() * scaling_factor
    eulers = df[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].to_numpy()
    lccmax = df['rlnLCCmax'].to_numpy() if 'rlnLCCmax' in df.columns else np.full(len(df), np.nan)

    distance_threshold_vox = distance_to_membrane_threshold / seg_apx
    min_r_vox = min_patch_radius / seg_apx
    max_r_vox = max_patch_radius / seg_apx
    fixed_r_vox = patch_radius / seg_apx
    step_vox = PATCH_RADIUS_SEARCH_STEP_A / seg_apx

    tree = cKDTree(membrane_coords_vox)
    nearest_dist, indices = tree.query(particle_coords_vox)
    closest_membrane_points = membrane_coords_vox[indices]

    #same per-particle logic as compute_all_tilts in cli.py, reusing its own building-block
    #functions - kept as a direct loop here (rather than calling compute_all_tilts itself) since
    #this needs the patch/normal geometry for image rendering, not just the summary numbers
    accepted = []
    for i in range(len(particle_coords_vox)):
        if nearest_dist[i] > distance_threshold_vox:
            continue
        pos = particle_coords_vox[i]
        nearest_pt = closest_membrane_points[i]
        rot, tilt, psi = eulers[i]
        particle_vector = euler_to_z_vector(rot, tilt, psi)

        if adaptive_patch_radius:
            candidate_idx = tree.query_ball_point(nearest_pt, r=max_r_vox)
            candidate_pts = membrane_coords_vox[candidate_idx]
            candidate_dists = np.linalg.norm(candidate_pts - nearest_pt, axis=1)
            result = find_stable_patch_radius(candidate_pts, candidate_dists, nearest_pt, pos,
                                               particle_vector, min_r_vox, max_r_vox, step_vox,
                                               angular_gap_threshold)
            if result is None or not result['valid']:
                continue
            patch, normal, gap, radius_vox = result['patch'], result['normal'], result['gap'], result['radius']
        else:
            patch_idx = tree.query_ball_point(nearest_pt, r=fixed_r_vox)
            if len(patch_idx) < 3:
                continue
            patch = membrane_coords_vox[patch_idx]
            raw_normal = calculate_patch_normal(patch)
            normal = resolve_normal_sign(raw_normal, patch, pos)
            gap = calculate_angular_gap(patch, nearest_pt, normal)
            if gap > angular_gap_threshold:
                continue
            radius_vox = fixed_r_vox

        angle = calculate_tilt_angle(particle_vector, normal)
        if angle > max_tilt_angle:
            continue
        curvature = calculate_local_curvature(patch, nearest_pt, normal)

        accepted.append(dict(
            idx=i, pos=pos, normal=normal, particle_vector=particle_vector, angle=angle, gap=gap,
            radius_A=radius_vox * seg_apx, curvature=curvature,
            lccmax=(None if np.isnan(lccmax[i]) else float(lccmax[i])),
        ))

    typer.echo(f"[{stem}] {len(accepted)}/{len(df)} particles pass the automatic criteria.")
    if not accepted:
        return []

    typer.echo(f"[{stem}] loading tomogram density and rendering review images...")
    with mrcfile.open(tomo_set['tomogram'], permissive=True) as mrc:
        tomo = np.asarray(mrc.data, dtype=np.float32)

    half_width_vox = (BOX_A / 2) / seg_apx
    grid_1d = np.linspace(-half_width_vox, half_width_vox, BOX_PIXELS)
    grid_b, grid_a = np.meshgrid(grid_1d, grid_1d, indexing='ij')

    context_half_width_vox = (CONTEXT_BOX_A / 2) / seg_apx
    context_grid_1d = np.linspace(-context_half_width_vox, context_half_width_vox, CONTEXT_BOX_PIXELS)
    context_grid_b, context_grid_a = np.meshgrid(context_grid_1d, context_grid_1d, indexing='ij')

    stem_cache_dir = cache_dir / stem
    stem_cache_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for rec in accepted:
        n_axis = rec['normal']
        pv = rec['particle_vector']
        tangential = pv - (pv @ n_axis) * n_axis
        tnorm = np.linalg.norm(tangential)
        h_axis = tangential / tnorm if tnorm > 1e-6 else build_tangent_frame(n_axis)[0]
        h_perp = np.cross(n_axis, h_axis)

        img_top = _extract_slice(tomo, rec['pos'], h_axis, h_perp, grid_a, grid_b)
        img_true = _extract_slice(tomo, rec['pos'], h_axis, n_axis, grid_a, grid_b)
        img_ortho = _extract_slice(tomo, rec['pos'], h_perp, n_axis, grid_a, grid_b)

        pv_top = np.array([pv @ h_axis, pv @ h_perp])
        pv_true = np.array([pv @ h_axis, pv @ n_axis])
        pv_ortho = np.array([pv @ h_perp, pv @ n_axis])

        filename = f"idx{rec['idx']}.png"
        _save_review_panel(stem_cache_dir / filename, img_top, img_true, img_ortho, pv_top, pv_true, pv_ortho)

        context_img = _extract_slice(tomo, rec['pos'], h_axis, n_axis, context_grid_a, context_grid_b)
        context_filename = f"idx{rec['idx']}_context.png"
        _save_context_panel(stem_cache_dir / context_filename, context_img, pv_true)

        records.append(dict(
            key=f"{stem}:{rec['idx']}", stem=stem, idx=int(rec['idx']),
            tilt=round(float(rec['angle']), 2), lcc=rec['lccmax'],
            curvature=round(float(rec['curvature']), 4), gap=round(float(rec['gap']), 2),
            radius_A=round(float(rec['radius_A']), 1), image=f"/api/image/{stem}/{filename}",
            context_image=f"/api/image/{stem}/{context_filename}",
        ))

    typer.echo(f"[{stem}] {len(records)} review images ready.")
    return records


def build_review_app(all_records: List[dict], cache_dir: Path, tomogram_starfiles: Dict[str, Path]):
    from flask import Flask, jsonify, request, send_from_directory, Response

    flask_app = Flask(__name__)
    decisions_path = cache_dir / "decisions.json"
    decisions: Dict[str, str] = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
    history: List[str] = []

    def save_decisions():
        tmp = decisions_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(decisions, indent=2))
        tmp.replace(decisions_path)

    @flask_app.route("/")
    def index():
        return Response(INDEX_HTML, mimetype="text/html")

    @flask_app.route("/api/queue")
    def queue():
        out = []
        for r in all_records:
            r2 = dict(r)
            r2["decision"] = decisions.get(r["key"])
            out.append(r2)
        return jsonify(out)

    @flask_app.route("/api/image/<stem>/<filename>")
    def image(stem, filename):
        return send_from_directory(cache_dir / stem, filename)

    @flask_app.route("/api/decision", methods=["POST"])
    def decision():
        data = request.get_json()
        key, value = data["key"], data.get("decision")
        if value is None:
            decisions.pop(key, None)
        else:
            decisions[key] = value
        history.append(key)
        save_decisions()
        return jsonify(_counts(decisions))

    @flask_app.route("/api/undo", methods=["POST"])
    def undo():
        if not history:
            return jsonify({"ok": False})
        key = history.pop()
        decisions.pop(key, None)
        save_decisions()
        return jsonify({"ok": True, "key": key, **_counts(decisions)})

    @flask_app.route("/api/export", methods=["POST"])
    def export():
        written = []
        for stem, starfile_path in tomogram_starfiles.items():
            keep_idxs = {r["idx"] for r in all_records if r["stem"] == stem and decisions.get(r["key"]) == "accept"}
            df, df_dict, block_name = load_star_data(starfile_path)
            out_df = df[df.index.isin(keep_idxs)].copy()
            out_data = {**df_dict, block_name: out_df} if block_name is not None else out_df
            out_path = starfile_path.parent / f"{stem}_reviewed.star"
            starfile.write(out_data, out_path, overwrite=True)
            written.append(dict(stem=stem, path=str(out_path), n=len(out_df)))

        n_unreviewed = sum(1 for r in all_records if decisions.get(r["key"]) is None)
        return jsonify({"written": written, "unreviewed_excluded": n_unreviewed})

    return flask_app


def _counts(decisions: Dict[str, str]) -> dict:
    return dict(accepted=sum(1 for v in decisions.values() if v == "accept"),
                rejected=sum(1 for v in decisions.values() if v == "reject"))


INDEX_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>gta review</title>
<style>
  :root {
    --bg: #14191a; --card: #1d2426; --line: #2c3537; --ink: #e7ecec; --ink-soft: #94a3a5;
    --teal: #3fa79c; --crimson: #d1495b; --amber: #f2a541;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--ink); font-family: -apple-system, "Segoe UI", sans-serif;
         margin: 0; height: 100vh; display: flex; flex-direction: column; }
  header { display: flex; justify-content: space-between; align-items: center;
           padding: 14px 24px; border-bottom: 1px solid var(--line); flex-shrink: 0; }
  header h1 { font-size: 15px; font-weight: 600; margin: 0; color: var(--ink-soft); letter-spacing: 0.02em; }
  #counts { font-family: ui-monospace, monospace; font-size: 13px; color: var(--ink-soft); }
  #counts b.acc { color: var(--teal); } #counts b.rej { color: var(--crimson); }
  main { flex: 1; display: flex; align-items: center; justify-content: center; min-height: 0; padding: 16px; gap: 28px; }
  #imgwrap { position: relative; border: 3px solid var(--line); border-radius: 10px; overflow: hidden;
             transition: border-color 0.1s; line-height: 0; background: #14191a; }
  #imgwrap.acc { border-color: var(--teal); } #imgwrap.rej { border-color: var(--crimson); }
  #img { max-height: 62vh; max-width: 58vw; display: block; }
  #badge { position: absolute; top: 10px; right: 10px; padding: 3px 10px; border-radius: 5px;
           font-size: 12px; font-weight: 600; letter-spacing: 0.03em; display: none; }
  #badge.acc { display: block; background: var(--teal); color: #06201d; }
  #badge.rej { display: block; background: var(--crimson); color: #2a0508; }
  #ctxcol { display: flex; flex-direction: column; align-items: center; gap: 8px; flex-shrink: 0; }
  #ctxwrap { border: 2px solid var(--line); border-radius: 8px; overflow: hidden; line-height: 0;
             background: #14191a; }
  #ctximg { max-height: 46vh; max-width: 26vw; display: block; }
  #ctxlabel { font-size: 11px; color: var(--ink-soft); letter-spacing: 0.05em; text-transform: uppercase; }
  #meta { width: 260px; font-size: 14px; line-height: 2.1; flex-shrink: 0; }
  #meta .row { display: flex; justify-content: space-between; border-bottom: 1px solid var(--line); padding: 2px 0; }
  #meta .label { color: var(--ink-soft); }
  #meta .val { font-family: ui-monospace, monospace; }
  #meta h2 { font-size: 20px; margin: 0 0 14px; font-weight: 600; }
  #progress-bar { height: 4px; background: var(--line); flex-shrink: 0; }
  #progress-fill { height: 100%; background: var(--teal); width: 0%; transition: width 0.15s; }
  footer { display: flex; justify-content: space-between; align-items: center; padding: 12px 24px;
           border-top: 1px solid var(--line); font-size: 12.5px; color: var(--ink-soft); flex-shrink: 0; }
  footer kbd { background: var(--card); border: 1px solid var(--line); border-radius: 4px; padding: 1px 6px;
               font-family: ui-monospace, monospace; color: var(--ink); margin: 0 2px; }
  button { background: var(--teal); color: #06201d; border: none; border-radius: 6px; padding: 8px 16px;
           font-size: 13px; font-weight: 600; cursor: pointer; }
  button:hover { opacity: 0.9; }
  #done { display: none; text-align: center; color: var(--teal); font-size: 15px; margin-top: 8px; }
</style>
</head>
<body>
<header>
  <h1>GTA REVIEW</h1>
  <div id="counts">accepted <b class="acc" id="c-acc">0</b> &middot; rejected <b class="rej" id="c-rej">0</b></div>
  <button onclick="doExport()">Export reviewed STAR files</button>
</header>
<div id="progress-bar"><div id="progress-fill"></div></div>
<main>
  <div id="imgwrap"><img id="img" src=""><div id="badge"></div></div>
  <div id="ctxcol">
    <div id="ctxwrap"><img id="ctximg" src=""></div>
    <div id="ctxlabel">context &middot; 2000&thinsp;&Aring;</div>
  </div>
  <div id="meta">
    <h2 id="m-title">-</h2>
    <div class="row"><span class="label">particle idx</span><span class="val" id="m-idx">-</span></div>
    <div class="row"><span class="label">tilt angle</span><span class="val" id="m-tilt">-</span></div>
    <div class="row"><span class="label">LCCmax</span><span class="val" id="m-lcc">-</span></div>
    <div class="row"><span class="label">curvature</span><span class="val" id="m-curv">-</span></div>
    <div class="row"><span class="label">coverage gap</span><span class="val" id="m-gap">-</span></div>
    <div class="row"><span class="label">patch radius</span><span class="val" id="m-radius">-</span></div>
    <div id="done">All particles reviewed.</div>
  </div>
</main>
<footer>
  <span><kbd>space</kbd> accept &nbsp; <kbd>del</kbd>/<kbd>backspace</kbd> reject &nbsp;
        <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> navigate &nbsp; <kbd>z</kbd> undo</span>
  <span id="progress-text">0 / 0</span>
</footer>
<script>
let queue = [];
let idx = 0;
let counts = {accepted: 0, rejected: 0};

async function init() {
  const res = await fetch('/api/queue');
  queue = await res.json();
  recomputeCounts();
  idx = queue.findIndex(r => !r.decision);
  if (idx === -1) idx = 0;
  render();
}

function recomputeCounts() {
  counts.accepted = queue.filter(r => r.decision === 'accept').length;
  counts.rejected = queue.filter(r => r.decision === 'reject').length;
  document.getElementById('c-acc').innerText = counts.accepted;
  document.getElementById('c-rej').innerText = counts.rejected;
}

function render() {
  if (queue.length === 0) { document.getElementById('m-title').innerText = 'No particles to review.'; return; }
  const r = queue[idx];
  document.getElementById('img').src = r.image;
  document.getElementById('ctximg').src = r.context_image;
  document.getElementById('m-title').innerText = r.stem;
  document.getElementById('m-idx').innerText = r.idx;
  document.getElementById('m-tilt').innerText = r.tilt.toFixed(1) + '°';
  document.getElementById('m-lcc').innerText = (r.lcc === null ? '-' : r.lcc.toFixed(2));
  document.getElementById('m-curv').innerText = r.curvature.toFixed(3);
  document.getElementById('m-gap').innerText = r.gap.toFixed(1) + '°';
  document.getElementById('m-radius').innerText = r.radius_A.toFixed(0) + ' A';
  document.getElementById('progress-text').innerText = (idx + 1) + ' / ' + queue.length;
  document.getElementById('progress-fill').style.width = (100 * (idx + 1) / queue.length) + '%';

  const wrap = document.getElementById('imgwrap');
  const badge = document.getElementById('badge');
  wrap.className = r.decision === 'accept' ? 'acc' : (r.decision === 'reject' ? 'rej' : '');
  badge.className = wrap.className;
  badge.innerText = r.decision === 'accept' ? 'ACCEPTED' : (r.decision === 'reject' ? 'REJECTED' : '');

  document.getElementById('done').style.display = queue.every(r => r.decision) ? 'block' : 'none';
  prefetch();
}

function prefetch() {
  for (let k = 1; k <= 3; k++) {
    if (queue[idx + k]) {
      const im = new Image(); im.src = queue[idx + k].image;
      const ctxIm = new Image(); ctxIm.src = queue[idx + k].context_image;
    }
  }
}

async function decide(value) {
  if (queue.length === 0) return;
  const r = queue[idx];
  r.decision = value;
  recomputeCounts();
  fetch('/api/decision', {method: 'POST', headers: {'Content-Type': 'application/json'},
                          body: JSON.stringify({key: r.key, decision: value})});
  if (idx < queue.length - 1) idx++;
  render();
}

function navigate(delta) {
  idx = Math.max(0, Math.min(queue.length - 1, idx + delta));
  render();
}

async function undo() {
  const res = await fetch('/api/undo', {method: 'POST'});
  const data = await res.json();
  if (data.ok) {
    const r = queue.find(x => x.key === data.key);
    if (r) r.decision = null;
    recomputeCounts();
    idx = queue.findIndex(x => x.key === data.key);
    render();
  }
}

async function doExport() {
  const res = await fetch('/api/export', {method: 'POST'});
  const data = await res.json();
  let msg = data.written.map(w => `${w.stem}: ${w.n} particles -> ${w.path}`).join('\n');
  if (data.unreviewed_excluded > 0) {
    msg += `\n\n${data.unreviewed_excluded} unreviewed particle(s) were NOT included.`;
  }
  alert(msg);
}

document.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); decide('accept'); }
  else if (e.code === 'Backspace' || e.code === 'Delete') { e.preventDefault(); decide('reject'); }
  else if (e.code === 'ArrowRight') navigate(1);
  else if (e.code === 'ArrowLeft') navigate(-1);
  else if (e.key === 'z' || e.key === 'Z') undo();
});

init();
</script>
</body>
</html>"""


@app.command()
def review(
    data_dir: Annotated[Path, typer.Argument(
        help="Directory containing tomograms/, segmentations/, and starfiles/ subdirectories - "
             "matching files (same stem) across all three are treated as one tomogram to review")],
    seg_apx: Annotated[float, typer.Option("--seg_apx", help="Segmentation/tomogram pixel size in Angstroms/pixel")],
    particles_apx: Annotated[
        float, typer.Option("--particles_apx", help="Particle coordinate pixel size in Angstroms/pixel")],
    patch_radius: Annotated[
        float, typer.Option("--patch_radius",
                             help="Fixed patch radius in Angstroms - only used with "
                                  "--no-adaptive-patch-radius")] = 150.0,
    min_patch_radius: Annotated[
        float, typer.Option("--min-patch-radius", help="Adaptive search minimum, in Angstroms")] = 40.0,
    max_patch_radius: Annotated[
        float, typer.Option("--max-patch-radius", help="Adaptive search maximum, in Angstroms")] = 150.0,
    adaptive_patch_radius: Annotated[
        bool, typer.Option("--adaptive-patch-radius/--no-adaptive-patch-radius",
                            help="Same adaptive patch radius selection as gta analyze-tilts")] = True,
    distance_to_membrane_threshold: Annotated[
        float, typer.Option("--distance-to-membrane-threshold", help="See gta analyze-tilts --help")] = 85.0,
    angular_gap_threshold: Annotated[
        float, typer.Option("--angular-gap-threshold", help="See gta analyze-tilts --help")] = 40.0,
    max_tilt_angle: Annotated[
        float, typer.Option("--max-tilt-angle", help="See gta analyze-tilts --help")] = 80.0,
    port: Annotated[int, typer.Option("--port", help="Local port to serve the review UI on")] = 5050,
):
    """
    Launch a local web app to rapidly accept/reject particles by eye, using real tomogram density
    slices oriented to each particle's own fitted membrane normal. Precomputes everything up
    front so the review itself has no lag - point a browser at it (directly, or via an SSH tunnel
    if running on a remote cluster) and use space/backspace to triage.
    """
    if not data_dir.is_dir():
        typer.echo(f"Input Error: {data_dir} is not a directory", err=True)
        raise typer.Exit(code=1)

    #must be absolute: Flask's send_from_directory resolves a relative `directory` against the
    #app's root_path (the installed gta package location), not the process cwd
    data_dir = data_dir.resolve()

    if seg_apx <= 0 or particles_apx <= 0:
        typer.echo("Input Error: --seg_apx and --particles_apx must be positive numbers", err=True)
        raise typer.Exit(code=1)

    try:
        tomo_sets = discover_tomogram_sets(data_dir)
    except (FileNotFoundError, ValueError) as e:
        typer.echo(f"Input Error: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(tomo_sets)} tomogram(s) to review: {[s['stem'] for s in tomo_sets]}")

    cache_dir = data_dir / ".gta_review_cache"
    cache_dir.mkdir(exist_ok=True)

    all_records: List[dict] = []
    tomogram_starfiles: Dict[str, Path] = {}
    for tomo_set in tomo_sets:
        records = analyze_tomogram_for_review(
            tomo_set, cache_dir, seg_apx, particles_apx,
            distance_to_membrane_threshold, angular_gap_threshold, max_tilt_angle,
            adaptive_patch_radius, patch_radius, min_patch_radius, max_patch_radius,
        )
        all_records.extend(records)
        tomogram_starfiles[tomo_set["stem"]] = tomo_set["starfile"]

    if not all_records:
        typer.echo("No particles passed the automatic criteria in any tomogram - nothing to review.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"\n{len(all_records)} particles ready for review across {len(tomo_sets)} tomogram(s).")

    flask_app = build_review_app(all_records, cache_dir, tomogram_starfiles)

    typer.echo(f"\nStarting review server on port {port}.")
    typer.echo("If this is running on a remote cluster, from your local machine run:")
    typer.echo(f"  ssh -L {port}:localhost:{port} <user>@<cluster-host>")
    typer.echo(f"then open http://localhost:{port} in your browser. Otherwise just open that "
               f"address directly.\n")

    flask_app.run(host="127.0.0.1", port=port, debug=False)
