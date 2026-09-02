#!/usr/bin/env python
"""Focused, annotated fine-grained nucleosome-dyad comparison at CTCF sites.

Uses the deep MNase Input dyad map (the reliable one) and the per-site binding
strengths already computed by run_overlay.py. Produces a clean 2x2:
  row 1: full +/-1000 bp   row 2: zoom +/-500 bp
  col 1: CTCF peak vs random   col 2: high vs low binding CTCF
with detected dyad peaks marked and +1 / NDR / NRL annotated.

Run after run_overlay.py (needs out/per_site_binding_strength.tsv) and
40_dyad_fine.sbatch (needs the mono-nucleosome dyad bigWig).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dimelo import track_overlay as tov

FORK = Path("/scratch/users/ngamarra/ngamarra_dimelo_fork")
DATA = FORK / "dimelo" / "test" / "output"
DEMO = FORK / "examples" / "h3_ctcf_overlay"
SCR = Path("/scratch/users/ngamarra/hiref_h3_mnase_chip/bigwig")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dyad-bw", default=str(SCR / "Input.chm13v2.dyad_mono.5bp.cpm.bw")
    )
    ap.add_argument(
        "--peak-bed", default=str(FORK / "dimelo/test/data/ctcf_demo_peak.bed")
    )
    ap.add_argument("--random-bed", default=str(DATA / "random_sites.bed"))
    ap.add_argument(
        "--strength-tsv", default=str(DEMO / "out" / "per_site_binding_strength.tsv")
    )
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--bins", type=int, default=400)
    ap.add_argument("--zoom", type=int, default=500)
    ap.add_argument("--out", default=str(DEMO / "out" / "fig5_dyad_focus.png"))
    args = ap.parse_args()

    PAL = {
        "CTCF peak": "tab:red",
        "random": "0.5",
        "high binding": "tab:red",
        "low binding": "tab:blue",
    }

    # per-site binding strength -> high/low split
    st = pd.read_csv(args.strength_tsv, sep="\t")
    strength = dict(zip(st["region_id"], st["binding_strength"]))
    thr = st["binding_strength"].median()

    peak = tov.read_bigwig_windows(
        args.dyad_bw, args.peak_bed, window_size=args.window, n_bins=args.bins
    )
    rand = tov.read_bigwig_windows(
        args.dyad_bw, args.random_bed, window_size=args.window, n_bins=args.bins
    )

    def _concat(a, b):
        import copy

        o = copy.copy(a)
        o.matrix = np.vstack([a.matrix, b.matrix])
        for f in ("region_ids", "chromosomes", "starts", "ends", "strands"):
            setattr(
                o,
                f,
                np.concatenate([np.asarray(getattr(a, f)), np.asarray(getattr(b, f))]),
            )
        return o

    pr_tw = _concat(peak, rand)
    pr_grp = np.array(["CTCF peak"] * peak.n_regions + ["random"] * rand.n_regions)
    hilo = np.where(
        pd.Series(peak.region_ids).map(strength).fillna(0).to_numpy() >= thr,
        "high binding",
        "low binding",
    )

    def annotate(ax, phas, key):
        """Mark +1 dyad and print +1/NDR/NRL for the CTCF/high-binding group."""
        pr = phas.get(key)
        if pr is None:
            return
        if pr.plus1 is not None:
            ax.axvline(pr.plus1, color=PAL.get(key, "k"), ls="--", lw=0.8, alpha=0.7)
        txt = []
        if pr.plus1 is not None:
            txt.append(f"+1 dyad ≈ {pr.plus1:.0f} bp")
        if pr.ndr_width is not None:
            txt.append(f"NDR ≈ {pr.ndr_width:.0f} bp")
        if pr.nrl_autocorr is not None:
            txt.append(f"NRL ≈ {pr.nrl_autocorr:.0f} bp")
        if txt:
            ax.text(
                0.02,
                0.97,
                "\n".join(txt),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
            )

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="row")
    for col, (tw, grp, mainkey, ttl) in enumerate(
        [
            (pr_tw, pr_grp, "CTCF peak", "Input MNase dyads: CTCF peak vs random"),
            (peak, hilo, "high binding", "Input MNase dyads: high vs low binding CTCF"),
        ]
    ):
        _, ph = tov.plot_dyad_phasing(
            tw, groups=grp, smooth_bp=15, ax=axes[0][col], palette=PAL, title=ttl
        )
        annotate(axes[0][col], ph, mainkey)
        # zoom row reuses the same profiles/peaks
        tov.plot_dyad_phasing(
            tw,
            groups=grp,
            smooth_bp=15,
            ax=axes[1][col],
            palette=PAL,
            title=f"{ttl.split(':')[1].strip()} (zoom ±{args.zoom} bp)",
        )
        axes[1][col].set_xlim(-args.zoom, args.zoom)
    fig.suptitle(
        "Fine-grained nucleosome dyad positioning relative to CTCF (mono-nucleosome, 5 bp)",
        y=1.0,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
