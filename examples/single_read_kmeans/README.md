# Single-Read KMeans Pipeline for DiMeLo-seq / m6A Footprinting

Cluster individual sequencing reads by their m6A methylation footprint patterns
around motif centers, produce publication-quality per-cluster pileup figures, and
compute per-cluster feature importances.

The pipeline runs **every registered BAM × every region-set** combination
automatically, extracting basemod data from BAMs into HDF5, QC-filtering reads,
building a rich feature matrix, running KMeans over a range of `k`, and emitting
figures + feature-importance reports.

---

## Table of Contents

- [Overview](#overview)
- [Installation / Dependencies](#installation--dependencies)
- [Inputs (all required)](#inputs-all-required)
- [Quick Start](#quick-start)
- [Subcommands](#subcommands)
- [Command-Line Options](#command-line-options)
- [HDF5 Extraction (automatic by default)](#hdf5-extraction-automatic-by-default)
- [Output Layout](#output-layout)
- [How It Works](#how-it-works)
  - [QC Filtering](#qc-filtering)
  - [Feature Overview](#feature-overview)
  - [Clustering & Label Caching](#clustering--label-caching)
  - [Feature Importance](#feature-importance)
- [The Paired Figure](#the-paired-figure)
- [Setting Pileup Y-Axis Bounds](#setting-pileup-y-axis-bounds)
- [Feature Reference](#feature-reference)
- [Tips & Notes](#tips--notes)
- [FAQ / Troubleshooting](#faq--troubleshooting)

---

## Overview

For each **BAM** ("tag") and each **region set** ("class"), the pipeline:

1. **Extracts** basemod calls from the BAM into an HDF5 file (once, cached,
   automatic by default).
2. **Loads** per-read windows (`±W/2` bp around each motif center), oriented by
   region strand.
3. **QC-filters** reads by callable-base and methylation thresholds.
4. **Featurizes** each read (PCA components, autocorrelation, window densities,
   run-length metrics, and ~12 read-level summary metrics).
5. **Clusters** reads with KMeans for a range of `k` (labels are cached to disk).
6. **Plots** a combined figure per `k`: side bars, single-read scatter, per-cluster
   mean profiles (with fraction-A overlay), and a cluster-share pie.
7. **Computes** per-cluster feature importances (heatmaps + JSON + TSV).

**Data flow:**
`BAM → (extract) → HDF5 → load windows → QC filter → featurize → KMeans (cached) → [paired plots] and/or [feature importance]`

---

## Installation / Dependencies

Python 3 with:

```bash
pip install numpy matplotlib h5py scikit-learn
```

Plus these project-specific packages (must be importable in your environment):

- `dimelo` (provides `load_processed` and `parse_bam`)
- `kmeans_interp` (provides `KMeansInterp` in `kmeans_interp.kmeans_feature_imp`)

`matplotlib` is used in headless mode (`Agg`), so no display is required.

---

## Inputs (all required)

There are **no built-in default paths** — you must register your inputs on the CLI.

| Input | Flag | Description |
|-------|------|-------------|
| **Output root** | `--base PATH` | Root directory for all outputs (required). |
| **Reference FASTA** | `--fasta PATH` | Single genome reference (required). |
| **BAM(s)** | `--bam NAME=PATH` | Aligned/sorted/indexed reads with MM/ML basemod tags. Repeatable; **at least one required**. |
| **Region BED(s)** | `--regions NAME=PATH` | One BED per "class". Repeatable; **at least one required**. |
| **Motif(s)** | `--motifs "A,0"` | e.g. `A,0` (adenine, index 0 = m6A). Default `A,0`. Results namespaced per motif. |

If any of `--base`, `--fasta`, at least one `--bam`, or at least one `--regions`
is missing, the pipeline exits with a clear error.

> **Region BED note:** The **motif center** is the midpoint of each interval
> (`(start + end) // 2`), e.g. a CTCF motif summit. Strand (column 6) is used to
> orient reads 5'→3' when `ORIENT=True`. Relative BED paths are resolved against
> `<base>/<bed-subdir>`; absolute paths are used as-is.

---

## Quick Start

Assuming the script is saved as `single_read_kmeans.py`:

```bash
# Just the plots (HDF5 built automatically if missing):
python single_read_kmeans.py \
  --base /my/out --fasta /my/genome.fasta \
  --bam sampleA=/my/A.bam --regions HH=/my/HH.bed \
  paired-plots --k-min 2 --k-max 10

# Just the feature importance:
python single_read_kmeans.py \
  --base /my/out --fasta /my/genome.fasta \
  --bam sampleA=/my/A.bam --regions HH=/my/HH.bed \
  feat-importance --k-min 2 --k-max 10

# Both plots AND feature importance:
python single_read_kmeans.py \
  --base /my/out --fasta /my/genome.fasta \
  --bam sampleA=/my/A.bam --regions HH=/my/HH.bed \
  all --k-min 2 --k-max 10
```

### Multiple BAMs / region sets

```bash
python single_read_kmeans.py \
  --base /my/out --fasta /my/genome.fasta \
  --bam sampleA=/my/A.bam --bam sampleB=/my/B.bam \
  --regions HH=/my/HH.bed --regions LL=/my/LL.bed \
  --motifs "A,0" \
  all --k-min 2 --k-max 10
```

This runs the full 2×2 grid (each BAM × each region set) automatically.

---

## Subcommands

The subcommand you choose *is* the choice of what to produce:

| Subcommand | Produces |
|------------|----------|
| `paired-plots` | **ONLY** the per-cluster pileup figures. |
| `feat-importance` | **ONLY** the per-cluster feature-importance reports. |
| `all` | **BOTH** paired-plots and feat-importance. |
| `build-h5` | **ONLY** extract combined-basemods HDF5 for the selected BAM(s), then stop. |
| `replot-feat-importance` | Re-render FI heatmaps from existing JSON (no recompute). |

---

## Command-Line Options

### Global options (before the subcommand)

| Option | Default | Description |
|--------|---------|-------------|
| `--base PATH` | **required** | Root output/data directory. |
| `--fasta PATH` | **required** | Reference FASTA. |
| `--bam NAME=PATH` | **≥1 required** | Register a BAM (repeatable). |
| `--regions NAME=PATH` | **≥1 required** | Register a region set (repeatable). |
| `--motifs M [M ...]` | `A,0` | Motif(s); results namespaced per motif. |
| `--bed-subdir NAME` | `intersections_150bp` | Subdir under `--base` for relative BED paths. |
| `--min-callable N` | `20` | Min callable A positions per read. |
| `--min-methyl N` | `5` | Min methylated A positions per read. |
| `--min-callable-each-side N` | `5` | Min callable A on each side of center. |
| `--scale-features` / `--no-scale-features` | on | Z-score features before clustering. |
| `--pileup-ymin FLOAT` | `None` (→ `0.0`) | Lower bound of per-cluster m6A y-axis. |
| `--pileup-ymax FLOAT` | `None` (→ auto) | Upper bound of per-cluster m6A y-axis. |
| `--only-bams NAME ...` | all | Restrict to specific BAM(s). |
| `--only-regions NAME ...` | all | Restrict to specific region set(s). |
| `--auto-build-h5` | **on** | Extract missing/corrupt HDF5 before analysis. |
| `--no-auto-build-h5` | — | Disable auto-extract; only use existing HDF5. |
| `--extract-cores N` | `16` | Cores for extraction. |

### Subcommand options

- `paired-plots`, `feat-importance`, `replot-feat-importance`, `all`:
  `--k-min` (default `2`), `--k-max` (default `15`)
- `paired-plots`, `all`: `--overwrite-labels`
- `feat-importance`: `--skip-existing`
- `all`: `--skip-existing-fi`

---

## HDF5 Extraction (automatic by default)

Extraction of the combined-basemods HDF5 from each BAM is **automatic**:

- Analysis commands (`paired-plots`, `feat-importance`, `all`) build any
  missing/corrupt HDF5 before running.
- Extraction is **cached**: an existing, valid HDF5 is reused; a truncated/corrupt
  one is detected and re-extracted.
- Pass `--no-auto-build-h5` to disable this — the pipeline then only runs on
  pre-existing HDF5 files and **skips** BAMs whose HDF5 is missing.
- The dedicated `build-h5` subcommand pre-stages HDF5 for all selected BAMs
  without running any analysis (useful on a big compute node before a batch run).

```bash
# Pre-extract everything, then analyze later:
python single_read_kmeans.py --base /my/out --fasta /my/genome.fasta \
  --bam sampleA=/my/A.bam --regions HH=/my/HH.bed \
  build-h5

# Analyze using only existing HDF5 (no extraction):
python single_read_kmeans.py ... --no-auto-build-h5 all
```

---

## Output Layout

Given `--base BASE` and motif slug `MS` (e.g. `A-0`):

```
BASE/
└── single_reads/
    ├── all_sites.union.bed                    # concatenation of all region sets
    ├── extracts/
    │   └── <MS>/
    │       └── <bam_tag>/reads.combined_basemods.h5
    ├── single_reads_kmeans/
    │   └── <MS>/<bam_tag>/<class>/
    │       ├── kNN_labels.scaled.npz          # cached labels (+ scaler)
    │       ├── kNN_paired.png
    │       └── kNN_paired.pdf
    └── kmeans_feature_importance_<MS>/
        └── <bam_tag>/<class>/
            ├── kNN_feat_importance.json
            ├── kNN_feat_importance.png / .pdf
            └── kNN_feat_importance_top.tsv
```

`NN` is the zero-padded value of `k` (e.g. `k05`).

---

## How It Works

### QC Filtering

A read is kept only if **all** of the following hold (defaults shown):

- callable A positions total ≥ `--min-callable` (20)
- methylated A positions total ≥ `--min-methyl` (5)
- callable A on the **left** half ≥ `--min-callable-each-side` (5)
- callable A on the **right** half ≥ `--min-callable-each-side` (5)

If fewer than `k_max + 1` reads survive QC for a class, that class is skipped.

### Feature Overview

Each read is turned into a feature vector composed of five blocks (full details
in the [Feature Reference](#feature-reference)):

1. **PCA components** of the raw m6A window (`n_pca_eff = min(N_PCA=8, n_reads, W)`).
2. **Autocorrelation** at lags `67, 70, 93, 167, 210` (nucleosome periodicities).
3. **Window densities** at ±25, ±37, ±50, ±150, ±250 bp.
4. **Run-length metrics** (contiguity/asymmetry of methylation runs).
5. **Read-level metrics**: `B, D, C_50, C_100, C_250, Delta_250, cm, R, R50,
   R80, Hnorm, dmin, S_100`.

Reads with any non-finite feature are dropped. When `--scale-features` is on
(default), features are z-scored so they contribute equally during clustering.

### Clustering & Label Caching

- KMeans (`n_init=10`, fixed `random_state=42`).
- Clusters are **relabeled by size** (cluster `0` = largest) for consistent
  colors/labels across `k` and runs.
- Labels are cached per `(tag, class, k)` as `.scaled.npz` (or `.raw.npz`), along
  with the scaler. Reruns reuse the cache when the read count matches, so
  **changing plot bounds does not require re-clustering**.

### Feature Importance

- Uses `KMeansInterp` with the **cached labels/centers**, so importances describe
  exactly the same clusters shown in the plots.
- Default method: `wcss_min` on the **raw** (unscaled) feature matrix.
- Outputs per `k`: a JSON (full importances), a top-feature TSV, and a
  row-normalized heatmap (color = |weight|/row-max; text = raw weight).
- `replot-feat-importance` regenerates heatmaps from existing JSON without
  recomputation.

---

## The Paired Figure

Each `kNN_paired.png/pdf` has four panels:

1. **Side bar** — reads grouped by cluster, colored and labeled with the cluster 
   number only (e.g. `C0`).
2. **Single-read scatter** — every kept read (subsampled for large classes),
   5'→3' region-oriented, m6A positions plotted as dots. Per-cluster counts and
   percentages are available in the titles.
3. **Per-cluster profiles** — smoothed mean m6A per cluster (left axis) with the
   overall class mean (dashed) and a **fraction-A overlay** (right axis, fixed
   0–1). SEM shaded.
4. **Pie chart** — cluster share of the class, with a legend.

Subsampling caps: `MAX_PER_CLUSTER=1500`, `MAX_TOTAL=6000`. Smoothing window:
`SMOOTH_W=30` bp.

---

## Setting Pileup Y-Axis Bounds

The **left (m6A) y-axis** of the per-cluster profile panels can be fixed for
consistent comparisons across figures. The right "fraction A" axis is always
0–1 and is not affected.

- **Default (unchanged behavior):** lower bound `0.0`, upper bound auto
  (`1.15×` the max smoothed profile).
- Override with `--pileup-ymin` and/or `--pileup-ymax`.
- If only one is given, the other keeps its default.
- Inverted/degenerate bounds (`ymax <= ymin`) trigger a warning and revert the
  upper bound to auto.

```bash
# Fix to [0, 0.15]
python single_read_kmeans.py ... --pileup-ymin 0 --pileup-ymax 0.15 paired-plots

# Only cap the top; keep the bottom at 0
python single_read_kmeans.py ... --pileup-ymax 0.2 all
```

Because labels are cached, re-running with new bounds only re-renders figures.

---

## Feature Reference

All features are computed from two aligned matrices of shape `(N_reads, W=2000)`:

- **`M`** (mod vector): nonzero where a base is called methylated.
- **`V`** (val vector): nonzero where a base is callable (an adenine that could
  be measured).

Two masks are derived per read: `m = (M>0)&(V>0)` (methylated **and** callable)
and `v = (V>0)` (callable). The coordinate axis `x = [-1000 … 999]` is centered
on the motif (`x=0` = summit; negative = upstream, positive = downstream after
orientation). `B = m.sum()` is total methylation per read.

### Block 1 — PCA components (`PC1 … PCn`)

Principal components of the raw methylation matrix `M`. Captures the dominant
*shapes* of methylation profiles across reads. `n_pca_eff = min(8, n_reads, W)`.
Abstract (not directly interpretable), but powerful for clustering.

### Block 2 — Autocorrelation (`ac_lag67, ac_lag70, ac_lag93, ac_lag167, ac_lag210`)

$$\text{ac}(L) = \frac{\sum_i x_i\, x_{i+L}}{\sum_i x_i^2},\quad x = v - \bar v$$

Similarity of the signal to itself shifted by `L` bp, normalized to ~[-1, 1].
Elevated values at these lags indicate **periodic / phased nucleosome arrays**
(~167 bp ≈ nucleosome repeat; ~147 bp ≈ wrapped DNA; smaller lags ≈ sub-nucleosomal).

### Block 3 — Window densities (`dens_w50, dens_w75, dens_w100, dens_w300, dens_w500`)

Mean of `M` over concentric windows of ±25, ±37, ±50, ±150, ±250 bp about the
center (name = full width). A multi-scale readout of how sharply methylation is
concentrated at the motif.

### Block 4 — Run-length features

A "run" is a maximal stretch of consecutive methylated positions in `m`.

| Feature | Meaning |
|---------|---------|
| `modA_run_n` | Number of runs (fragmentation). |
| `modA_run_maxlen` | Length (bp) of the longest run. |
| `modA_run_long_center` | Longest run's center, normalized to [0,1] (0=upstream end, 1=downstream end). |
| `modA_run_long_edge` | Longest run's distance from the middle, normalized (0=centered, 1=edge). |
| `modA_run_pos_wsum` | Σ (run length × signed center bp) — directional, mass-weighted asymmetry. |
| `modA_run_pos_maxabs` | The single most extreme (length × off-center) run. |

### Block 5 — Per-read metrics

| Feature | Formula (per read) | Meaning |
|---------|--------------------|---------|
| `B` | `m.sum()` | Total methylated bases ("footprint mass"). |
| `D` | `B / v.sum()` | Overall methylation density (methyl / callable). |
| `C_50`, `C_100`, `C_250` | `m[\|x\|≤r].sum() / B` | Fraction of the read's methylation within ±r bp (central concentration). |
| `Delta_250` | `Dcenter − Ddist` | Central (±250) density minus distal (500–1000) density; central enrichment contrast. |
| `cm` | `(x·m).sum() / B` | Center of mass of methylation (bp; sign = up/downstream shift). |
| `R` | `sqrt( Σ m·(x−cm)² / B )` | RMS spread (std-dev) about the center of mass. |
| `R50`, `R80` | radius holding 50% / 80% of methylation | Quantile-based compactness. |
| `Hnorm` | `−Σ p·log p / log K`, K=40 bins | Normalized spatial entropy (1=uniform, 0=focal). |
| `dmin` | `min(\|x\|)` over methylated positions | Distance from center to nearest methylation. |
| `S_100` | `Ctrue − median(shifted)` | Centering score vs. ±{200,400,600,800} offset windows. |

### Summary

| Block | Captures | Interpretable? |
|-------|----------|----------------|
| PCA | Dominant profile shapes | No (abstract) |
| Autocorrelation | Nucleosome periodicity / phasing | Yes |
| Window densities | Multi-scale central methylation | Yes |
| Run features | Contiguity & asymmetry of footprints | Yes |
| Per-read metrics | Amount, concentration, spread, centering | Yes (most) |

Together these describe **how much** methylation there is (`B`, `D`), **where**
it is (`cm`, `C_r`, `dmin`, run positions), **how concentrated/spread** it is
(`R`, `R50`, `R80`, `Hnorm`), **how it contrasts with background** (`Delta_250`,
`S_100`), and **whether it's periodic** (autocorrelation).

**Column order** (as produced by `build_feature_names`):

```
PC1..PCn, ac_lag67, ac_lag70, ac_lag93, ac_lag167, ac_lag210,
dens_w50, dens_w75, dens_w100, dens_w300, dens_w500,
modA_run_n, modA_run_maxlen, modA_run_long_center, modA_run_long_edge,
modA_run_pos_wsum, modA_run_pos_maxabs,
B, D, C_50, C_100, C_250, Delta_250, cm, R, R50, R80, Hnorm, dmin, S_100
```

---

## Tips & Notes

- **Reproducibility:** all randomness uses `random_state=42`.
- **Namespacing:** extracts and results are separated per motif slug, so you can
  run multiple motifs without collisions.
- **Reusing HDF5:** valid HDF5 files are reused; truncated/corrupt files are
  detected and re-extracted.
- **Subsetting runs:** use `--only-bams` and/or `--only-regions` to limit the
  grid.
- **Editing hard-coded params:** thresholds, window size `W`, `N_PCA`, palette,
  smoothing, and metric radii are constants near the top of the script.

---

## FAQ / Troubleshooting

**"No output root provided" / "No BAMs provided" / "No region sets provided" / "No reference FASTA provided"**
You must supply `--base`, `--fasta`, at least one `--bam`, and at least one
`--regions`. There are no built-in defaults.

**"no readable h5 for tag=... (remove --no-auto-build-h5 or run build-h5 first)"**
You disabled auto-extraction and no HDF5 exists. Remove `--no-auto-build-h5`, or
pre-extract with the `build-h5` subcommand.

**"SKIP: only N reads after QC"**
Too few reads survived QC for that class at your `k_max`. Lower `--k-max`, relax
QC thresholds, or check that the BED region overlaps enough coverage.

**"h5 not found ... tag '...' is not a registered BAM"**
The tag you're analyzing isn't a known BAM. Register it with
`--bam NAME=/path.bam`.

**Feature-importance heatmaps look empty / all `·0`**
Weights below `5e-3` render as `·0`. Inspect the JSON/TSV for exact values, or
try a more informative `k`.

**Changing y-bounds didn't recluster**
That's intended — labels are cached. Use `--overwrite-labels` (paired-plots/all)
if you truly want to refit.
