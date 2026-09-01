#!/usr/bin/env python
"""Overlay HuRef H3 MNase nucleosome-occupancy tracks on clustered/classified
targeting CTCF sites (DiMeLo one-pot barcode17), using dimelo.track_overlay.

Pipeline (mirrors tutorial_targeting.ipynb section 9, then overlays chromatin):
  1. Extract single-read 6mA windows over CTCF peaks (bound) + random sites
     (unbound) for the targeting dataset.
  2. Site-wise classifier (peak vs random, peak-methylation-density features) ->
     per-read P(bound) and per-site binding strength (fraction of reads bound).
  3. Sample the H3 bigWig tracks over the same centered windows.
  4. Figures:
       fig1_h3_peak_vs_random.png  - H3 metaprofile at CTCF peaks vs random
       fig2_overlay.png            - H3 (top) over 6mA pileup of bound vs unbound reads (bottom)
       fig3_heatmap_binding.png    - H3 log2 heatmap at CTCF sites sorted by binding strength

All coordinates are CHM13v2.0 (barcode BAMs and H3 tracks share the assembly),
so no liftover is needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dimelo import cluster
from dimelo import track_overlay as tov

# --------------------------------------------------------------------------- #
# Defaults (Sherlock layout). Override on the CLI if paths move.
# --------------------------------------------------------------------------- #
FORK = Path("/scratch/users/ngamarra/ngamarra_dimelo_fork")
OUT = FORK / "dimelo" / "test" / "output"
DATA = FORK / "dimelo" / "test" / "data"
BWDIR = Path("/oak/stanford/groups/altemose/ngamarra/analyses/huref_h3_mnase_chip/bigwig")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="barcode17", help="targeting dataset name (default barcode17)")
    p.add_argument("--hdf5", default=None, help="per-read HDF5 (default OUT/<dataset>_extract/reads.combined_basemods.h5)")
    p.add_argument("--peak-bed", default=str(DATA / "ctcf_demo_peak.bed"))
    p.add_argument("--random-bed", default=str(OUT / "random_sites.bed"))
    p.add_argument("--dyad-bw", default=str(BWDIR / "H3.chm13v2.MNase_dyad.cpm.bw"))
    p.add_argument("--log2-bw", default=str(BWDIR / "H3_vs_Input.chm13v2.log2.bw"))
    # Fine-grained mono-nucleosome dyad tracks (5 bp bins, 120-180 bp fragments).
    # Default to the scratch bigwig dir where 40_dyad_fine.sbatch writes them.
    SCR = Path("/scratch/users/ngamarra/hiref_h3_mnase_chip/bigwig")
    p.add_argument("--dyad-mono-input", default=str(SCR / "Input.chm13v2.dyad_mono.5bp.cpm.bw"))
    p.add_argument("--dyad-mono-h3", default=str(SCR / "H3.chm13v2.dyad_mono.5bp.cpm.bw"))
    p.add_argument("--dyad-bins", type=int, default=400, help="bins across the window for dyad phasing (default 400 = 5 bp)")
    p.add_argument("--motif", default="A,0", help="modification motif for the read pileup (default 6mA)")
    p.add_argument("--window", type=int, default=1000, help="half-window bp (default 1000)")
    p.add_argument("--min-valid-fraction", type=float, default=0.05)
    p.add_argument("--outdir", default=str(FORK / "examples" / "h3_ctcf_overlay" / "out"))
    return p


def extract(hdf5, bed, motif, win):
    """Extract per-read windows over one region set."""
    return cluster.extract_read_windows(
        hdf5_file=str(hdf5), motifs=[motif], regions=str(bed),
        config=cluster.ReadWindowExtractionConfig(window_size=win, orientation_aware=True),
        span_full_window=False,
    )


def main() -> None:
    args = build_argparser().parse_args()
    win = args.window
    hdf5 = args.hdf5 or str(OUT / f"{args.dataset}_extract" / "reads.combined_basemods.h5")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    PAL = {"bound": "tab:red", "unbound": "0.5", "high binding": "tab:red",
           "low binding": "tab:blue", "CTCF peak": "tab:red", "random": "0.5"}

    # --- 1. extract bound (peak) + unbound (random) reads --------------------
    print("[1] extracting single-read 6mA windows over peak + random ...")
    rws, src = [], []
    for bed, lab in [(args.peak_bed, "bound"), (args.random_bed, "unbound")]:
        rws.append(extract(hdf5, bed, args.motif, win))
        src.append(lab)  # one label per result; merge broadcasts to that result's reads
    rw_all = cluster.merge_read_window_results(rws, source_labels=src, align="error")

    # --- 2. features + site-wise classifier (peak vs random) -----------------
    print("[2] features + XGBoost bound/unbound classifier ...")
    mvf = args.min_valid_fraction
    feat, _ = cluster.read_window_feature_matrix(
        rw_all, n_pca=6, use_peak_features=True, require_nonzero_valid=True, min_valid_fraction=mvf,
    )
    # Reproduce the exact read mask feature extraction applied, to keep matrices aligned.
    vv = np.asarray(rw_all.val_matrix).sum(axis=1)
    mv = vv > 0
    if rw_all.val_matrix.shape[1] > 0:
        mv &= (vv / rw_all.val_matrix.shape[1]) >= mvf
    meta = pd.DataFrame(rw_all.metadata).loc[mv].reset_index(drop=True)
    X = np.asarray(rw_all.data_matrix)[mv]
    V = np.asarray(rw_all.val_matrix)[mv]
    labels = meta["source_label"].to_numpy()  # 'bound'/'unbound'
    span = X.shape[1]
    read_pos = np.arange(-win, win) if span == 2 * win else np.linspace(-win, win, span)

    clf = cluster.classify_read_features_binary(feat, sample_labels=labels, classifier="xgboost", random_state=42)
    print("    test roc_auc=%.3f acc=%.3f"
          % (clf["metrics"]["test"]["roc_auc"], clf["metrics"]["test"]["accuracy"]))

    # per-read P(bound), oriented so proba == P(bound)
    pr = clf["predictions"].copy()
    orient_pos = pr.loc[pr["pred_label"] == "bound", "proba"].ge(0.5).mean() > 0.5
    pr["p_bound"] = pr["proba"] if orient_pos else 1 - pr["proba"]
    p_bound = np.full(X.shape[0], np.nan)
    pred_lab = np.array([""] * X.shape[0], dtype=object)
    p_bound[pr["row_index"].to_numpy()] = pr["p_bound"].to_numpy()
    pred_lab[pr["row_index"].to_numpy()] = pr["pred_label"].to_numpy()

    # peak reads only (the "targeting CTCF" reads) for the read pileup
    is_peak = labels == "bound"
    peak_call = np.where(p_bound[is_peak] >= 0.5, "bound", "unbound")

    # per-site binding strength (fraction of a CTCF site's reads called bound)
    site = tov.site_labels_from_reads(meta[is_peak], (p_bound[is_peak] >= 0.5).astype(float), agg="mean")
    site = site.rename(columns={"value": "binding_strength"})
    print(f"    {is_peak.sum()} peak reads over {len(site)} CTCF sites; "
          f"median binding strength = {site['binding_strength'].median():.2f}")

    # --- 3. sample H3 tracks over the same windows ---------------------------
    print("[3] sampling H3 bigWig windows ...")
    dyad_peak = tov.read_bigwig_windows(args.dyad_bw, args.peak_bed, window_size=win, n_bins=200)
    dyad_rand = tov.read_bigwig_windows(args.dyad_bw, args.random_bed, window_size=win, n_bins=200)
    log2_peak = tov.read_bigwig_windows(args.log2_bw, args.peak_bed, window_size=win, n_bins=200)
    log2_rand = tov.read_bigwig_windows(args.log2_bw, args.random_bed, window_size=win, n_bins=200)
    # per-bp track aligned to the read matrix columns for the overlay
    log2_peak_bp = tov.read_bigwig_windows(args.log2_bw, args.peak_bed, window_size=win, n_bins=span)

    # --- fig1: H3 at CTCF peaks vs random ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, (tp, tr, name) in zip(
        axes, [(dyad_peak, dyad_rand, "H3 MNase dyad (CPM)"), (log2_peak, log2_rand, "H3/Input log2")]
    ):
        for tw, lab in [(tp, "CTCF peak"), (tr, "random")]:
            mp = tov.metaprofile(tw, smooth_bp=25)
            ax.plot(mp["position"], mp["mean"], label=f"{lab} (n={tw.n_regions})", color=PAL[lab])
            ax.fill_between(mp["position"], mp["lo"], mp["hi"], color=PAL[lab], alpha=0.2, lw=0)
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("position rel. to site center (bp)")
        ax.set_ylabel(name)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("HuRef H3 nucleosome occupancy at DiMeLo CTCF sites vs random")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_h3_peak_vs_random.png", dpi=150)
    plt.close(fig)

    # --- fig2: overlay H3 over bound/unbound read pileup ---------------------
    # site groups: split CTCF sites by binding strength (median) for the H3 panel
    site_ids_peak = (meta.loc[is_peak, "chromosome"].astype(str) + ":"
                     + meta.loc[is_peak, "region_start"].astype(int).astype(str) + "-"
                     + meta.loc[is_peak, "region_end"].astype(int).astype(str))
    thr = site["binding_strength"].median()
    strength_map = site["binding_strength"].to_dict()
    # align site groups to the per-bp track rows (log2_peak_bp region order)
    tw_ids = pd.Series(log2_peak_bp.region_ids)
    site_group = np.where(tw_ids.map(strength_map).fillna(0).to_numpy() >= thr, "high binding", "low binding")

    # read pileup uses only the CTCF-peak reads, grouped by their bound/unbound call
    X_peak, V_peak = X[is_peak], V[is_peak]
    fig = tov.overlay_track_with_reads(
        log2_peak_bp, X_peak, val_matrix=V_peak,
        read_groups=peak_call,
        site_groups=site_group,
        read_positions=log2_peak_bp.positions,
        mod_label=f"fraction methylated ({args.motif})",
        track_label="H3/Input log2",
        smooth_bp=25, palette=PAL,
        title="H3 occupancy vs single-molecule binding at CTCF sites",
    )
    fig.savefig(outdir / "fig2_overlay.png", dpi=150)
    plt.close(fig)

    # --- fig3: H3 log2 heatmap sorted by per-site binding strength -----------
    strength_vec = tw_ids.map(strength_map).fillna(np.nan).to_numpy()
    fig, (axh, axb) = plt.subplots(
        1, 2, figsize=(8, 6), gridspec_kw={"width_ratios": [4, 1], "wspace": 0.05}, sharey=True
    )
    tov.plot_track_heatmap(log2_peak_bp, sort_by=strength_vec, smooth_bp=25, ax=axh,
                           title="H3/Input log2 at CTCF sites (sorted by binding strength)")
    order = np.argsort(strength_vec)[::-1]
    axb.barh(np.arange(len(order)), strength_vec[order], color="tab:red", height=1.0)
    axb.set_xlim(0, 1)
    axb.set_ylim(len(order), 0)
    axb.set_xlabel("binding strength")
    axb.set_yticks([])
    fig.tight_layout()
    fig.savefig(outdir / "fig3_heatmap_binding.png", dpi=150)
    plt.close(fig)

    # --- fig4: fine-grained nucleosome dyad phasing relative to CTCF ----------
    import copy
    import os

    def _concat(a, b):
        out = copy.copy(a)  # TrackWindows is a plain dataclass -> shallow copy + set
        out.matrix = np.vstack([a.matrix, b.matrix])
        for f in ("region_ids", "chromosomes", "starts", "ends", "strands"):
            setattr(out, f, np.concatenate([np.asarray(getattr(a, f)), np.asarray(getattr(b, f))]))
        return out

    if os.path.exists(args.dyad_mono_input):
        print("[4] fine-grained dyad phasing (mono-nucleosome dyads) ...")
        nb = args.dyad_bins
        din_peak = tov.read_bigwig_windows(args.dyad_mono_input, args.peak_bed, window_size=win, n_bins=nb)
        din_rand = tov.read_bigwig_windows(args.dyad_mono_input, args.random_bed, window_size=win, n_bins=nb)
        dids = pd.Series(din_peak.region_ids)
        dgroup = np.where(dids.map(strength_map).fillna(0).to_numpy() >= thr, "high binding", "low binding")

        has_h3 = os.path.exists(args.dyad_mono_h3)
        rows = 2 if has_h3 else 1
        fig, axes = plt.subplots(rows, 2, figsize=(13, 3.6 * rows), squeeze=False)
        pr_grp = np.array(["CTCF peak"] * din_peak.n_regions + ["random"] * din_rand.n_regions)
        _, ph1 = tov.plot_dyad_phasing(_concat(din_peak, din_rand), groups=pr_grp, smooth_bp=15,
            ax=axes[0][0], palette=PAL, title="Input MNase dyads: CTCF peak vs random")
        _, ph2 = tov.plot_dyad_phasing(din_peak, groups=dgroup, smooth_bp=15,
            ax=axes[0][1], palette=PAL, title="Input MNase dyads: high vs low binding CTCF")
        phas_all = {"input_peak_vs_random": ph1, "input_high_vs_low": ph2}
        if has_h3:
            dh3_peak = tov.read_bigwig_windows(args.dyad_mono_h3, args.peak_bed, window_size=win, n_bins=nb)
            dh3_rand = tov.read_bigwig_windows(args.dyad_mono_h3, args.random_bed, window_size=win, n_bins=nb)
            h3_grp = np.array(["CTCF peak"] * dh3_peak.n_regions + ["random"] * dh3_rand.n_regions)
            _, ph3 = tov.plot_dyad_phasing(_concat(dh3_peak, dh3_rand), groups=h3_grp, smooth_bp=20,
                ax=axes[1][0], palette=PAL, title="H3 ChIP dyads: CTCF peak vs random")
            _, ph4 = tov.plot_dyad_phasing(dh3_peak, groups=dgroup, smooth_bp=20,
                ax=axes[1][1], palette=PAL, title="H3 ChIP dyads: high vs low binding CTCF")
            phas_all["h3_peak_vs_random"] = ph3
            phas_all["h3_high_vs_low"] = ph4
        fig.tight_layout()
        fig.savefig(outdir / "fig4_dyad_phasing.png", dpi=150)
        plt.close(fig)

        recs = []
        for panel, d in phas_all.items():
            for g, pr in d.items():
                recs.append({"panel": panel, "group": g, "plus1_bp": pr.plus1, "minus1_bp": pr.minus1,
                             "ndr_width_bp": pr.ndr_width, "nrl_peaks_bp": pr.nrl_peaks,
                             "nrl_autocorr_bp": pr.nrl_autocorr, "n_dyad_peaks": int(pr.peak_positions.size)})
        met = pd.DataFrame(recs)
        met.to_csv(outdir / "dyad_phasing_metrics.tsv", sep="\t", index=False)
        print("    dyad phasing metrics:")
        print(met.to_string(index=False))
    else:
        print(f"[4] SKIP dyad phasing (missing {args.dyad_mono_input}); run 40_dyad_fine.sbatch first")

    # --- summary table -------------------------------------------------------
    site.reset_index().to_csv(outdir / "per_site_binding_strength.tsv", sep="\t", index=False)
    print(f"[done] figures + table written to {outdir}")


if __name__ == "__main__":
    main()
