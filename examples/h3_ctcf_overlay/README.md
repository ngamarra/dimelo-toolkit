# H3 MNase overlay on targeting CTCF sites

Overlay a continuous chromatin track (HuRef **H3 MNase-seq** nucleosome occupancy,
re-aligned to CHM13v2.0) onto the CTCF sites analysed in `tutorial_targeting.ipynb`,
aligned with the DiMeLo single-read 6mA classification of **bound vs unbound** reads.

Both the DiMeLo one-pot BAMs (barcode17/18) and the H3 bigWigs are on **CHM13v2.0**,
so the overlay needs no liftover.

## Core capability — `dimelo.track_overlay`

Reusable, dataset-agnostic module (`dimelo/track_overlay.py`):

| function | purpose |
|---|---|
| `read_bigwig_windows(bw, regions, window_size=, n_bins=, orientation_aware=)` | sample a bigWig over `±window` windows centered on regions, same frame as `cluster.extract_read_windows` |
| `TrackWindows` | `(n_regions, n_positions)` signal container (`.subset`, `.mean_profile`, `.to_frame`) |
| `metaprofile(tw, groups=, smooth_bp=)` | collapse regions → mean ± SEM profile, optionally by group |
| `site_labels_from_reads(metadata, values, agg=)` | aggregate a per-read value (e.g. P(bound)) to a per-site value |
| `read_pileup(data_matrix, val_matrix, rows=)` | valid-weighted per-position methylation fraction |
| `plot_track_metaprofile` / `plot_track_heatmap` | grouped line plot / row-sorted heatmap |
| `overlay_track_with_reads(tw, data_matrix, read_groups=, site_groups=)` | **headline**: track metaprofile stacked over the read pileup on a shared axis |

`read_bigwig_windows` uses the *same centering and `-`-strand flip* as
`extract_read_windows(orientation_aware=True)`, so bigWig columns line up 1:1 with the
single-read methylation matrix.

## Nucleosome dyad phasing (fine-grained)

`dimelo.track_overlay` also resolves *where nucleosome dyads sit relative to CTCF*:

| function | purpose |
|---|---|
| `nucleosome_phasing(tw, groups=, smooth_bp=)` | detect the phased dyad peaks from a dyad metaprofile → `PhasingResult` (+1/−1 nucleosome offsets, NDR width, nucleosome repeat length from peak spacing **and** autocorrelation) |
| `plot_dyad_phasing(tw, groups=)` | fine dyad metaprofile with detected peaks marked; returns `(ax, {group: PhasingResult})` |

This needs a **fine, mono-nucleosome-gated dyad track**. Build one from a BAM with
`../../` `40_dyad_fine.sbatch` style call:
`bamCoverage --MNase --minFragmentLength 120 --maxFragmentLength 180 --binSize 5 --ignoreDuplicates`.
The deep **MNase Input** is the clean high-resolution nucleosome map; the H3 ChIP here
(~6.7M unique pairs) is too shallow for reliable single-dyad calling and is only
confirmatory — prefer Input (or a deeper MNase) for dyad positioning.

## Run

Fine dyad tracks first (writes `Input/H3.chm13v2.dyad_mono.5bp.cpm.bw` to the scratch
bigwig dir), then the demo:

```bash
sbatch /scratch/users/ngamarra/hiref_h3_mnase_chip/40_dyad_fine.sbatch   # mono-nucleosome dyad tracks
cd /scratch/users/ngamarra/ngamarra_dimelo_fork
sbatch examples/h3_ctcf_overlay/run_overlay.sbatch          # unit tests + demo (figs 1-4)
# override any path/param, e.g.:
sbatch examples/h3_ctcf_overlay/run_overlay.sbatch --dataset barcode18 --window 1500
```

Requires the fork env `/home/groups/altemose/envs/ngamarra_dimelo_fork` (has `pyBigWig`).
Fig4 is skipped gracefully if the dyad tracks are absent.

## Outputs (`out/`)

- `fig1_h3_peak_vs_random.png` — H3 dyad + H3/Input-log2 metaprofile at CTCF peaks vs random
  (expect a nucleosome-depleted / phased signature at bound CTCF).
- `fig2_overlay.png` — H3/Input log2 metaprofile (top, sites split by binding strength) over the
  6mA methylation pileup of **bound vs unbound** single reads (bottom), shared x-axis.
- `fig3_heatmap_binding.png` — per-site H3/Input log2 heatmap sorted by binding strength, with a
  binding-strength sidebar.
- `fig4_dyad_phasing.png` — fine (5 bp, mono-nucleosome) dyad metaprofiles with detected nucleosome
  peaks: Input & H3 dyads, CTCF-peak-vs-random and high-vs-low-binding. Deep Input shows the CTCF
  NDR + phased flanking dyads (+1 ≈ +170 bp, NRL ≈ 165–190 bp); random has a central dyad.
- `per_site_binding_strength.tsv` — per-CTCF-site fraction of reads called bound.
- `dyad_phasing_metrics.tsv` — +1/−1 dyad offsets, NDR width, and NRL per panel/group.

## Reuse on your own data

```python
from dimelo import cluster, track_overlay as tov
rw = cluster.extract_read_windows(hdf5, ["A,0"], regions="sites.bed",
        config=cluster.ReadWindowExtractionConfig(window_size=1000, orientation_aware=True))
tw = tov.read_bigwig_windows("mytrack.bw", "sites.bed", window_size=1000, n_bins=rw.data_matrix.shape[1])
tov.overlay_track_with_reads(tw, rw.data_matrix, val_matrix=rw.val_matrix,
        read_groups=my_read_labels, site_groups=my_site_labels)
```
Any bigWig works (ATAC, ChIP, conservation, ...); any per-read grouping (cluster id, class, haplotype).
