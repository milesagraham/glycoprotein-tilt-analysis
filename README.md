# GTA: Glycoprotein Tilt Analysis

A command-line tool for calculating the tilt angle of viral surface glycoproteins
relative to an irregularly segmented membrane, from a RELION-style STAR file of
particle coordinates/orientations and a segmentation volume.

For each particle it finds the nearest point on the segmented membrane, fits a
local plane to estimate the membrane normal there, and measures the angle
between that normal and the particle's own orientation.

![Example: one particle's local membrane patch, fitted normal, and orientation vector, viewed side-on](docs/example_idx51_sideview.png)

## Installation

Create a conda environment and `pip install` the package into it:

```bash
conda create -n gta python=3.10
conda activate gta
pip install -e .
```

This installs the `gta` command and its dependencies (`typer`, `mrcfile`, `starfile`, `numpy`, `scipy`,
`pandas`, `flask`, `matplotlib`). Python 3.8+ works; 3.10 is just a safe default. Re-run `pip install -e .`
after pulling new changes, and `conda activate gta` again at the start of every new shell session.

## Usage

`gta` has two subcommands: `analyze-tilts` (compute tilt angles) and `review` (interactively
triage the results). Run `gta --help` to see both.

```bash
gta analyze-tilts segmentation.mrc particles.star -11-seg_apx 7.456 --particles_apx 3.728
```

`seg_file` and `star_file` are positional; pixel sizes are required since the
segmentation and particle coordinates are often binned differently. Everything
else has a sensible default.

### Key options

| Option | Default | What it does |
|---|---|---|
| `--adaptive-patch-radius` / `--no-adaptive-patch-radius` | on | Per-particle: search for the smallest patch radius at which the tilt angle has settled, instead of using one fixed radius for everyone. |
| `--min-patch-radius`, `--max-patch-radius` | 40, 150 Å | Search range used when adaptive selection is on. |
| `--patch_radius` | 150 Å | Fixed patch radius, used only with `--no-adaptive-patch-radius`. |
| `--distance-to-membrane-threshold` | 85 Å | Particles farther than this from any segmented membrane are flagged as implausible picks (roughly the expected ectodomain height). |
| `--angular-gap-threshold` | 40° | Particles whose local patch has a wide empty angular sector (segmentation edge, missing-wedge dropout) are flagged as unreliable. |
| `--max-tilt-angle` | 80° | Particles tilted more than this are flagged as implausible (a real membrane-anchored glycoprotein can't tilt close to 90° without its ectodomain clashing into the membrane - typically junk picks, e.g. on segmented ice contamination). |
| `--output` / `-o` | `tilts_output.star` | Output path. |

Run `gta analyze-tilts --help` for the full list.

## Output

Two STAR files are written:

- `<output>.star` — every input particle, with five new columns:
  - `rlnOriginalIndex` — the particle's row position in the input STAR file, so a specific particle can be found again later (in this output, or in ArtiaX).
  - `rlnMembraneTiltAngle` — the result, in degrees. `NaN` for particles excluded by any of the checks below.
  - `rlnMembraneCurvature` — sign/size of the local membrane curvature relative to the particle (negative = convex/expected, positive = concave, i.e. the particle appears to sit inside the membrane rather than on it).
  - `rlnAngularCoverageGap` — widest empty angular sector (degrees) found around the particle's patch, recorded even for excluded particles so it's clear why.
  - `rlnPatchRadiusUsed` — the patch radius (Å) actually used for that particle.
- `<output>_accepted_coordinates.star` — the same, with excluded (`NaN`) particles dropped, ready to load directly into tools like ArtiaX that can't filter on `NaN`.

## Reviewing results: `gta review`

The four automatic criteria above catch most problems, but not everything (e.g. a particle
sitting on segmented ice contamination can still pass all four). `gta review` launches a local
web app for fast, keyboard-driven manual triage on top of the automatically-accepted particles,
showing a real tomogram slice through each pick oriented to its own fitted membrane normal.

```bash
gta review /path/to/data_dir --seg_apx 7.456 --particles_apx 3.728
```

`data_dir` must contain three subdirectories - `tomograms/`, `segmentations/`, and `starfiles/` -
and files that share the same stem (filename without extension) across all three are treated as
one tomogram to review. Every particle from every matched tomogram is combined into a single
review queue. All the same threshold options as `analyze-tilts` are available (run
`gta review --help` for the full list).

Everything expensive (the adaptive patch-radius search, tomogram slicing, image rendering) runs
once up front, before the server starts, so the review itself has no lag. Once it's running:

- Open the printed URL directly, or - if running on a remote cluster - tunnel it first:
  ```bash
  ssh -L 5050:localhost:5050 user@cluster-host
  ```
  then open `http://localhost:5050` locally.
- `space` accepts the current particle and advances; `Backspace`/`Delete` rejects it (junk) and
  advances; `←`/`→` navigate without deciding; `z` undoes the last decision.
- Decisions are saved continuously to `data_dir/.gta_review_cache/decisions.json`, so a review
  session can be closed and resumed later without losing progress.
- The "Export reviewed STAR files" button writes one `<stem>_reviewed.star` per tomogram
  (accepted particles only) next to that tomogram's input STAR file.

## Example

The image above is particle from a real tomogram (shown here at a fixed
100 Å patch radius for a clearer view of the membrane; the tool's own adaptive
selection would typically use a smaller radius for this particle), viewed
side-on to the fitted normal: the grey cloud is its local membrane patch, the
black cross is the nearest membrane point, the blue arrow is the fitted
membrane normal, and the red arrow is the particle's own orientation — the
angle between them (here, 14.6°) is what gets reported. 
