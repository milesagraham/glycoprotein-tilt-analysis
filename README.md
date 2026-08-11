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

`gta` is a two-step, batch-first tool: `analyze-tilts` always computes tilt angles for every
tomogram found under a directory, and `review` is a strictly separate second step that only ever
displays results `analyze-tilts` already computed - it never runs any analysis itself. Run
`gta --help` to see both commands.

### Directory layout

Both commands take one `data_dir` argument, expected to contain these subdirectories, with
matching files (same filename stem, e.g. `ts_028.mrc` / `ts_028.star`) across them treated as one
tomogram:

- `segmentations/` — one segmentation `.mrc` per tomogram. Always required.
- `starfiles/` — one RELION-style particle `.star` file per tomogram. Always required.
- `tomograms/` — one tomogram density `.mrc` per tomogram. Only required when using
  `--prepare-review` (see below) - plain tilt-angle calculation never touches the density volume,
  only the segmentation and particle coordinates.

### Step 1: `gta analyze-tilts`

For each particle, finds the nearest point on the segmented membrane, fits a local plane to
estimate the membrane normal there, and measures the angle between that normal and the particle's
own orientation. Runs across every matched tomogram under `data_dir`, in parallel:

```bash
gta analyze-tilts /path/to/data_dir --seg_apx 7.456 --particles_apx 3.728
```

Pixel sizes are required since the segmentation and particle coordinates are often binned
differently. Everything else has a sensible default.

#### Key options

| Option | Default | What it does |
|---|---|---|
| `--adaptive-patch-radius` / `--no-adaptive-patch-radius` | on | Per-particle: search for the smallest patch radius at which the tilt angle has settled, instead of using one fixed radius for everyone. |
| `--min-patch-radius`, `--max-patch-radius` | 40, 150 Å | Search range used when adaptive selection is on. |
| `--patch_radius` | 150 Å | Fixed patch radius, used only with `--no-adaptive-patch-radius`. |
| `--distance-to-membrane-threshold` | 85 Å | Particles farther than this from any segmented membrane are flagged as implausible picks (roughly the expected ectodomain height). |
| `--angular-gap-threshold` | 40° | Particles whose local patch has a wide empty angular sector (segmentation edge, missing-wedge dropout) are flagged as unreliable. |
| `--max-tilt-angle` | 80° | Particles tilted more than this are flagged as implausible (a real membrane-anchored glycoprotein can't tilt close to 90° without its ectodomain clashing into the membrane - typically junk picks, e.g. on segmented ice contamination). |
| `--workers` / `-j` | 4 | Number of tomograms analyzed in parallel - they're fully independent of each other. `--workers 1` disables parallelism. |
| `--prepare-review` | off | Also render the tomogram density images `gta review` needs (see below). Requires `tomograms/`. |

Run `gta analyze-tilts --help` for the full list.

`--workers` parallelizes across tomograms using Python's `ProcessPoolExecutor`, which only spreads
work across CPU cores on the single machine the command is running on - it cannot reach across
nodes. On a cluster, submit `gta analyze-tilts` (with or without `--prepare-review`) as a
**single-node** job, sized to that node's core count; requesting multiple SLURM nodes for one
invocation will not speed it up, since every node but the one actually running the process sits
idle.

#### Output

Two STAR files are written per tomogram to `data_dir/tilts_output/` (not into `starfiles/` itself,
so re-running doesn't pick its own previous outputs back up as spurious input):

- `<stem>_tilts_output.star` — every input particle, with five new columns:
  - `rlnOriginalIndex` — the particle's row position in the input STAR file, so a specific particle can be found again later (in this output, or in ArtiaX).
  - `rlnMembraneTiltAngle` — the result, in degrees. `NaN` for particles excluded by any of the checks below.
  - `rlnMembraneCurvature` — sign/size of the local membrane curvature relative to the particle (negative = convex/expected, positive = concave, i.e. the particle appears to sit inside the membrane rather than on it).
  - `rlnAngularCoverageGap` — widest empty angular sector (degrees) found around the particle's patch, recorded even for excluded particles so it's clear why.
  - `rlnPatchRadiusUsed` — the patch radius (Å) actually used for that particle.
- `<stem>_tilts_output_accepted_coordinates.star` — the same, with excluded (`NaN`) particles dropped, ready to load directly into tools like ArtiaX that can't filter on `NaN`.

### Step 2: reviewing results with `gta review`

The four automatic criteria above catch most problems, but not everything (e.g. a particle
sitting on segmented ice contamination can still pass all four). Add `--prepare-review` to
`analyze-tilts` to also render, for every automatically-accepted particle, a real tomogram slice
oriented to its own fitted membrane normal:

```bash
gta analyze-tilts /path/to/data_dir --seg_apx 7.456 --particles_apx 3.728 --prepare-review
```

This needs a `tomograms/` subdirectory (unlike plain `analyze-tilts`) and writes its renders to
`data_dir/gta_review_inputs/`. It's the expensive part of the whole tool (tomogram slicing, image
rendering), so it's parallelized the same way across `--workers`, and results are cached per
tomogram, fingerprinted on the input files and the options used - re-running only redoes tomograms
whose inputs or parameters actually changed, so an interrupted batch job can safely be resubmitted.

Once that's done, `gta review` launches a local web app for fast, keyboard-driven manual triage.
It only ever reads `data_dir/gta_review_inputs/` and serves it - it never computes anything itself,
and refuses to start if `--prepare-review` hasn't been run yet:

```bash
gta review /path/to/data_dir
```

Every particle from every prepared tomogram is combined into a single review queue.

#### Splitting compute from review on a cluster

Because `analyze-tilts --prepare-review` never starts a network server, and `gta review` never
computes anything, they're a natural fit for opposite ends of a cluster: run the parallel analysis
as a batch job on a compute node with a high `--workers`, then review on the login node:

```bash
# on a compute node, e.g. inside a Slurm job:
gta analyze-tilts /path/to/data_dir --seg_apx 7.456 --particles_apx 3.728 --workers 32 --prepare-review
```

```bash
# on the login node, once the batch job has finished:
gta review /path/to/data_dir
```

Once `gta review` is running:

- Open the printed URL directly, or - if running on a remote cluster - tunnel it first:
  ```bash
  ssh -L 5050:localhost:5050 user@cluster-host
  ```
  then open `http://localhost:5050` locally.
- `space` accepts the current particle and advances; `Backspace`/`Delete` rejects it (junk) and
  advances; `←`/`→` navigate without deciding; `z` undoes the last decision.
- Decisions are saved continuously to `data_dir/gta_review_inputs/decisions.json`, so a review
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
