"""Overlay continuous chromatin tracks (bigWig) on DiMeLo single-read windows.

This module bridges two coordinate frames used elsewhere in ``dimelo``:

* **Per-read modification windows** produced by :func:`dimelo.cluster.extract_read_windows`
  -- a ``(n_reads, span_bp)`` binary matrix of methylation calls, centered and
  orientation-normalised on a set of genomic *regions* (e.g. CTCF motif sites).
* **Per-locus continuous signal** stored in a bigWig -- e.g. MNase H3 nucleosome
  occupancy, coverage, or a log2(ChIP/Input) ratio.

Because both are anchored on the *same* region centers, we can sample the bigWig
over exactly the same ``+/- window_size`` frame (with the same orientation flip
for ``-`` strand regions) and plot the two directly on a shared x-axis. Typical
use is to ask *"how does nucleosome occupancy relate to single-molecule binding
at these sites?"* -- e.g. overlay an H3 metaprofile on the 6mA methylation pileup
of reads that were classified bound vs unbound.

The only hard dependency beyond the core stack is ``pyBigWig`` (import-guarded so
the rest of ``dimelo`` still imports if it is absent).

Public API
----------
``read_bigwig_windows``    sample a bigWig over centered region windows -> ``TrackWindows``
``TrackWindows``           dataclass holding the sampled ``(n_regions, n_pos)`` matrix
``metaprofile``            collapse regions -> mean (+/- error) profile, optionally by group
``site_labels_from_reads`` aggregate a per-read value to a per-region (site) value
``plot_track_metaprofile`` line plot of one/many grouped metaprofiles
``plot_track_heatmap``     regions x position heatmap, optionally row-sorted
``overlay_track_with_reads``  headline: track metaprofile stacked over the read pileup
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:  # Optional dependency -- only needed to read bigWig tracks.
    import pyBigWig

    _HAS_PYBIGWIG = True
except Exception:  # pragma: no cover - pyBigWig is optional
    pyBigWig = None
    _HAS_PYBIGWIG = False

try:  # Optional -- only needed for dyad peak detection in nucleosome_phasing.
    from scipy.signal import find_peaks

    _HAS_SCIPY_FIND_PEAKS = True
except Exception:  # pragma: no cover - SciPy is optional
    find_peaks = None
    _HAS_SCIPY_FIND_PEAKS = False


# --------------------------------------------------------------------------- #
# Region loading
# --------------------------------------------------------------------------- #
def _load_regions(regions: str | Path | pd.DataFrame | Sequence[Any]) -> pd.DataFrame:
    """Normalise many region spellings into a tidy DataFrame.

    Accepts a BED path, a list of ``(chrom, start, end[, ..., strand])`` tuples,
    or a DataFrame carrying ``chromosome/start/end`` (and optionally ``strand``).
    Returns columns ``chromosome, start, end, strand, region_id`` with ``start``/
    ``end`` as ints and ``region_id`` formatted ``chrom:start-end``.
    """
    if isinstance(regions, pd.DataFrame):
        df = regions.copy()
        # Tolerate a few common column spellings.
        ren = {
            "chrom": "chromosome",
            "chr": "chromosome",
            "chromStart": "start",
            "chromEnd": "end",
        }
        df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
        if "strand" not in df.columns:
            df["strand"] = "+"
    elif isinstance(regions, (str, Path)):
        rows = []
        with open(regions) as fh:
            for ln in fh:
                if not ln.strip() or ln.startswith(("#", "track", "browser")):
                    continue
                f = ln.rstrip("\n").split("\t")
                # BED: chrom start end [name score strand]; strand may be col 4 or 6.
                strand = "+"
                if len(f) >= 6 and f[5] in ("+", "-"):
                    strand = f[5]
                elif len(f) >= 4 and f[3] in ("+", "-"):
                    strand = f[3]
                rows.append((f[0], int(f[1]), int(f[2]), strand))
        df = pd.DataFrame(rows, columns=["chromosome", "start", "end", "strand"])
    else:  # sequence of tuples / lists
        rows = []
        for r in regions:
            chrom, start, end = r[0], int(r[1]), int(r[2])
            strand = r[-1] if len(r) > 3 and r[-1] in ("+", "-") else "+"
            rows.append((chrom, start, end, strand))
        df = pd.DataFrame(rows, columns=["chromosome", "start", "end", "strand"])

    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df["region_id"] = (
        df["chromosome"].astype(str)
        + ":"
        + df["start"].astype(str)
        + "-"
        + df["end"].astype(str)
    )
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Core container
# --------------------------------------------------------------------------- #
@dataclass
class TrackWindows:
    """Continuous-track signal sampled over centered region windows.

    Attributes
    ----------
    matrix : np.ndarray
        ``(n_regions, n_positions)`` signal. NaN where the track has no data
        (bigWig gaps or out-of-contig padding).
    positions : np.ndarray
        ``(n_positions,)`` bp offsets from region center. Orientation-aware:
        positive = downstream in the region's strand orientation. When ``n_bins``
        is used these are bin-center offsets instead of single bp.
    region_ids, chromosomes, starts, ends, strands : np.ndarray
        Per-region metadata aligned to matrix rows.
    track_name : str
    window_size : int
        Half-window in bp (full span = ``2 * window_size``).
    n_bins : int | None
        ``None`` for per-bp sampling; otherwise the number of bins per window.
    """

    matrix: np.ndarray
    positions: np.ndarray
    region_ids: np.ndarray
    chromosomes: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    strands: np.ndarray
    track_name: str
    window_size: int
    n_bins: int | None = None

    @property
    def n_regions(self) -> int:
        return self.matrix.shape[0]

    def subset(self, mask: Sequence[bool] | np.ndarray) -> TrackWindows:
        """Return a new TrackWindows keeping only rows where ``mask`` is True."""
        m = np.asarray(mask, dtype=bool)
        return TrackWindows(
            matrix=self.matrix[m],
            positions=self.positions,
            region_ids=np.asarray(self.region_ids)[m],
            chromosomes=np.asarray(self.chromosomes)[m],
            starts=np.asarray(self.starts)[m],
            ends=np.asarray(self.ends)[m],
            strands=np.asarray(self.strands)[m],
            track_name=self.track_name,
            window_size=self.window_size,
            n_bins=self.n_bins,
        )

    def mean_profile(self) -> np.ndarray:
        """NaN-aware mean signal across regions -> ``(n_positions,)``."""
        with warnings.catch_warnings():  # all-NaN columns are fine -> NaN
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(self.matrix, axis=0)

    def to_frame(self) -> pd.DataFrame:
        """Per-region metadata as a DataFrame (no signal matrix)."""
        return pd.DataFrame(
            {
                "region_id": self.region_ids,
                "chromosome": self.chromosomes,
                "start": self.starts,
                "end": self.ends,
                "strand": self.strands,
            }
        )


def _region_center(start: int, end: int, mode: str) -> int:
    if mode == "midpoint":
        return (start + end) // 2
    if mode == "start":
        return start
    if mode == "end":
        return end
    raise ValueError(f"center must be 'midpoint'|'start'|'end', got {mode!r}")


def read_bigwig_windows(
    bigwig: str | Path,
    regions: str | Path | pd.DataFrame | Sequence[Any],
    *,
    window_size: int = 1000,
    n_bins: int | None = None,
    center: str = "midpoint",
    orientation_aware: bool = True,
    aggregate: str = "mean",
    quiet: bool = True,
) -> TrackWindows:
    """Sample a bigWig track over ``+/- window_size`` windows centered on regions.

    Parameters
    ----------
    bigwig : path to a .bw/.bigWig file.
    regions : BED path, DataFrame, or list of tuples (see :func:`_load_regions`).
    window_size : half-window in bp. Full span sampled is ``2 * window_size``.
    n_bins : if given, summarise each window into ``n_bins`` bins (fast, smaller);
        otherwise sample per-bp (``2 * window_size`` columns).
    center : where to anchor the window on each region ('midpoint'|'start'|'end').
    orientation_aware : reverse the sampled vector for ``-`` strand regions so that
        positive positions are always downstream in region orientation (matches
        ``extract_read_windows(orientation_aware=True)``).
    aggregate : bigWig bin statistic when ``n_bins`` is set ('mean'|'max'|'min'|'sum').

    Returns
    -------
    TrackWindows
    """
    if not _HAS_PYBIGWIG:
        raise ImportError(
            "read_bigwig_windows requires pyBigWig. Install with `pip install pyBigWig`."
        )
    df = _load_regions(regions)
    bw = pyBigWig.open(str(bigwig))
    try:
        chrom_lens = bw.chroms()  # {chrom: length}
        # Symmetric inclusive window [c-w, c+w] -> width 2w+1. The odd width keeps
        # the center exactly representable, so the orientation flip (reverse) is an
        # exact palindrome around position 0 (no half-bp/1-bp shift on '-' strand).
        span = 2 * window_size + 1
        n_pos = n_bins if n_bins is not None else span
        mat = np.full((len(df), n_pos), np.nan, dtype=float)
        missing_chroms: set[str] = set()

        for i, row in df.iterrows():
            chrom = str(row["chromosome"])
            clen = chrom_lens.get(chrom)
            if clen is None:
                missing_chroms.add(chrom)
                continue
            c = _region_center(int(row["start"]), int(row["end"]), center)
            s, e = c - window_size, c + window_size + 1
            # Clamp to contig; remember where valid data lands within the window.
            cs, ce = max(0, s), min(clen, e)
            if cs >= ce:
                continue
            if n_bins is not None:
                # Bin only the in-bounds sub-window, then place into the full frame.
                nb = max(1, round(n_bins * (ce - cs) / span))
                vals = bw.stats(chrom, cs, ce, type=aggregate, nBins=nb)
                vals = np.array([np.nan if v is None else v for v in vals], dtype=float)
                # map bin centers of the sub-window onto the full-window bin grid
                sub_centers = cs + (np.arange(nb) + 0.5) * (ce - cs) / nb
                full_idx = np.clip(
                    ((sub_centers - s) / span * n_bins).astype(int), 0, n_bins - 1
                )
                mat[i, full_idx] = vals
            else:
                vals = np.array(bw.values(chrom, cs, ce), dtype=float)  # NaN for gaps
                mat[i, (cs - s) : (cs - s) + len(vals)] = vals
            if orientation_aware and row["strand"] == "-":
                mat[i, :] = mat[i, ::-1]

        if missing_chroms and not quiet:
            warnings.warn(
                f"{len(missing_chroms)} chromosome(s) absent from {Path(bigwig).name}: "
                f"{sorted(missing_chroms)[:5]}{'...' if len(missing_chroms) > 5 else ''}",
                RuntimeWarning,
                stacklevel=2,
            )
    finally:
        bw.close()

    if n_bins is not None:
        positions = np.linspace(-window_size, window_size, n_bins)
    else:
        positions = np.arange(-window_size, window_size + 1)

    return TrackWindows(
        matrix=mat,
        positions=positions,
        region_ids=df["region_id"].to_numpy(),
        chromosomes=df["chromosome"].to_numpy(),
        starts=df["start"].to_numpy(),
        ends=df["end"].to_numpy(),
        strands=df["strand"].to_numpy(),
        track_name=Path(bigwig).stem,
        window_size=window_size,
        n_bins=n_bins,
    )


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _smooth(v: np.ndarray, win_bp: int | None) -> np.ndarray:
    """Simple NaN-aware centered moving average (odd window in bp)."""
    if not win_bp or win_bp <= 1:
        return v
    k = int(win_bp) | 1  # force odd
    kernel = np.ones(k)
    filled = np.where(np.isnan(v), 0.0, v)
    counts = np.convolve((~np.isnan(v)).astype(float), kernel, mode="same")
    summed = np.convolve(filled, kernel, mode="same")
    with np.errstate(invalid="ignore"):
        out = summed / counts
    return out


def metaprofile(
    tw: TrackWindows,
    *,
    groups: Sequence[Any] | None = None,
    error: str | None = "sem",
    smooth_bp: int | None = None,
) -> pd.DataFrame:
    """Collapse regions into mean (+/- error) profiles, optionally split by group.

    Returns a long DataFrame with columns ``position, group, mean, lo, hi, n``.
    ``error`` is ``'sem'``, ``'std'`` or ``None``.
    """
    if groups is None:
        groups = np.array(["all"] * tw.n_regions)
    groups = np.asarray(groups)
    out = []
    for g in pd.unique(groups):
        sub = tw.matrix[groups == g]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(sub, axis=0)
            n = np.sum(~np.isnan(sub), axis=0)
            if error == "std":
                err = np.nanstd(sub, axis=0)
            elif error == "sem":
                err = np.nanstd(sub, axis=0) / np.sqrt(np.maximum(n, 1))
            else:
                err = np.zeros_like(mean)
        mean = _smooth(mean, smooth_bp)
        err = _smooth(err, smooth_bp)
        out.append(
            pd.DataFrame(
                {
                    "position": tw.positions,
                    "group": g,
                    "mean": mean,
                    "lo": mean - err,
                    "hi": mean + err,
                    "n": n,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def site_labels_from_reads(
    metadata: Sequence[dict] | pd.DataFrame,
    values: Sequence[Any],
    *,
    agg: str = "mean",
    region_key: tuple[str, ...] = ("chromosome", "region_start", "region_end"),
) -> pd.DataFrame:
    """Aggregate a per-read value to a per-site (region) value.

    Given the ``metadata`` from a :class:`ReadWindowExtractionResult` and a matching
    per-read ``values`` array (e.g. P(bound), or a cluster id), collapse to one value
    per region so it can be used to group/sort :class:`TrackWindows`.

    Returns a DataFrame indexed by ``region_id`` with a ``value`` (and ``n_reads``)
    column. ``agg`` is any pandas groupby-agg name ('mean','median','max',
    'first', ...); use 'first' for categorical labels.
    """
    md = (
        pd.DataFrame(list(metadata))
        if not isinstance(metadata, pd.DataFrame)
        else metadata.copy()
    )
    md = md.reset_index(drop=True)
    md["_value"] = np.asarray(values)
    md["region_id"] = (
        md[region_key[0]].astype(str)
        + ":"
        + md[region_key[1]].astype(int).astype(str)
        + "-"
        + md[region_key[2]].astype(int).astype(str)
    )
    g = md.groupby("region_id")["_value"].agg([agg, "size"])
    g.columns = ["value", "n_reads"]
    return g


# --------------------------------------------------------------------------- #
# Read-side pileup (methylation) helper
# --------------------------------------------------------------------------- #
def read_pileup(
    data_matrix: np.ndarray,
    val_matrix: np.ndarray | None = None,
    *,
    rows: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Per-position mean methylation across reads (a "pileup").

    ``data_matrix`` is ``(n_reads, span)`` binary calls, ``val_matrix`` marks valid
    positions (1) vs no-data (0). For a subset of ``rows`` (default all), returns
    ``sum(data*val)/sum(val)`` per column -- the fraction of *informative* reads
    that are methylated at each position.
    """
    d = np.asarray(data_matrix, dtype=float)
    if rows is not None:
        d = d[np.asarray(rows)]
    if val_matrix is None:
        return np.nanmean(d, axis=0)
    v = np.asarray(val_matrix, dtype=float)
    if rows is not None:
        v = v[np.asarray(rows)]
    num = np.nansum(d * v, axis=0)
    den = np.nansum(v, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_track_metaprofile(
    tw: TrackWindows,
    *,
    groups: Sequence[Any] | None = None,
    error: str | None = "sem",
    smooth_bp: int | None = None,
    ax=None,
    palette: dict | None = None,
    ylabel: str | None = None,
    title: str | None = None,
):
    """Line plot of grouped metaprofiles with a shaded error band."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 3.2))
    mp = metaprofile(tw, groups=groups, error=error, smooth_bp=smooth_bp)
    for g, sub in mp.groupby("group"):
        color = None if palette is None else palette.get(g)
        (line,) = ax.plot(
            sub["position"],
            sub["mean"],
            label=f"{g} (n={int(sub['n'].max())})",
            color=color,
        )
        if error:
            ax.fill_between(
                sub["position"],
                sub["lo"],
                sub["hi"],
                color=line.get_color(),
                alpha=0.2,
                lw=0,
            )
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("position relative to site center (bp)")
    ax.set_ylabel(ylabel or tw.track_name)
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_track_heatmap(
    tw: TrackWindows,
    *,
    sort_by: Sequence[float] | np.ndarray | None = None,
    ascending: bool = False,
    smooth_bp: int | None = None,
    ax=None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
):
    """Regions x position heatmap of the track, optionally row-sorted.

    ``sort_by`` is a per-region value (aligned to matrix rows) -- e.g. per-site
    binding strength -- so rows order by that quantity (largest first by default).
    """
    import matplotlib.pyplot as plt

    order = np.arange(tw.n_regions)
    if sort_by is not None:
        order = np.argsort(np.asarray(sort_by, dtype=float))
        if not ascending:
            order = order[::-1]
    mat = tw.matrix[order]
    if smooth_bp:
        mat = np.vstack([_smooth(r, smooth_bp) for r in mat])
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 6))
    if vmin is None:
        vmin = np.nanpercentile(mat, 2)
    if vmax is None:
        vmax = np.nanpercentile(mat, 98)
    im = ax.imshow(
        mat,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[tw.positions[0], tw.positions[-1], tw.n_regions, 0],
        interpolation="nearest",
    )
    ax.axvline(0, color="w", lw=0.8, ls=":")
    ax.set_xlabel("position relative to site center (bp)")
    ax.set_ylabel(f"sites (n={tw.n_regions})" + ("" if sort_by is None else ", sorted"))
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=tw.track_name)
    return ax


# --------------------------------------------------------------------------- #
# Nucleosome dyad phasing
# --------------------------------------------------------------------------- #
@dataclass
class PhasingResult:
    """Nucleosome-positioning summary from a dyad-density metaprofile.

    Attributes
    ----------
    positions, profile : the (smoothed) mean dyad profile the metrics come from.
    peak_positions : bp offsets of detected dyad peaks (the phased nucleosome array).
    plus1, minus1 : bp offset of the first dyad peak on the downstream / upstream
        side of center (the +1 / -1 nucleosome relative to the site center).
    ndr_width : distance between the -1 and +1 dyads (nucleosome-depleted region span).
    nrl_peaks : nucleosome repeat length from the median spacing of adjacent peaks.
    autocorr_lags, autocorr : autocorrelation of the profile (detrended).
    nrl_autocorr : NRL estimate = first autocorrelation peak lag (bp).
    """

    positions: np.ndarray
    profile: np.ndarray
    peak_positions: np.ndarray
    plus1: float | None
    minus1: float | None
    ndr_width: float | None
    nrl_peaks: float | None
    autocorr_lags: np.ndarray
    autocorr: np.ndarray
    nrl_autocorr: float | None


def _autocorr_nrl(
    profile: np.ndarray, bin_bp: float, min_lag_bp: float, max_lag_bp: float
):
    """Autocorrelation of a detrended profile; return (lags_bp, acf, first-peak-lag)."""
    v = np.asarray(profile, dtype=float)
    v = v[~np.isnan(v)]
    if v.size < 8:
        return np.array([]), np.array([]), None
    v = v - v.mean()
    full = np.correlate(v, v, mode="full")
    acf = full[full.size // 2 :]
    acf = acf / acf[0] if acf[0] != 0 else acf
    lags_bp = np.arange(acf.size) * bin_bp
    lo = max(1, int(round(min_lag_bp / bin_bp)))
    hi = min(acf.size, int(round(max_lag_bp / bin_bp)))
    nrl = None
    if hi > lo + 2:
        seg = acf[lo:hi]
        if _HAS_SCIPY_FIND_PEAKS:
            pk, _ = find_peaks(seg)
            if len(pk):
                nrl = float((lo + pk[0]) * bin_bp)
        if nrl is None:  # fallback: argmax of the segment
            nrl = float((lo + int(np.argmax(seg))) * bin_bp)
    return lags_bp, acf, nrl


def _phasing_from_profile(
    positions: np.ndarray,
    profile: np.ndarray,
    *,
    min_distance_bp: float = 120.0,
    prominence: float | None = None,
    min_lag_bp: float = 120.0,
    max_lag_bp: float = 1000.0,
) -> PhasingResult:
    """Detect dyad peaks + phasing metrics on a single mean profile."""
    positions = np.asarray(positions, dtype=float)
    profile = np.asarray(profile, dtype=float)
    bin_bp = float(np.median(np.diff(positions))) if positions.size > 1 else 1.0

    peak_pos = np.array([])
    if _HAS_SCIPY_FIND_PEAKS:
        filled = np.where(np.isnan(profile), np.nanmin(profile), profile)
        if prominence is None:  # scale-free default: a fraction of the dynamic range
            rng = np.nanmax(filled) - np.nanmin(filled)
            prominence = 0.05 * rng if rng > 0 else None
        dist = max(1, int(round(min_distance_bp / bin_bp)))
        idx, _ = find_peaks(filled, distance=dist, prominence=prominence)
        peak_pos = positions[idx]

    pos_peaks = np.sort(peak_pos[peak_pos > 0]) if peak_pos.size else np.array([])
    neg_peaks = np.sort(peak_pos[peak_pos < 0]) if peak_pos.size else np.array([])
    plus1 = float(pos_peaks[0]) if pos_peaks.size else None
    minus1 = float(neg_peaks[-1]) if neg_peaks.size else None
    ndr_width = (plus1 - minus1) if (plus1 is not None and minus1 is not None) else None
    nrl_peaks = (
        float(np.median(np.diff(np.sort(peak_pos)))) if peak_pos.size >= 2 else None
    )

    lags, acf, nrl_ac = _autocorr_nrl(profile, bin_bp, min_lag_bp, max_lag_bp)
    return PhasingResult(
        positions=positions,
        profile=profile,
        peak_positions=peak_pos,
        plus1=plus1,
        minus1=minus1,
        ndr_width=ndr_width,
        nrl_peaks=nrl_peaks,
        autocorr_lags=lags,
        autocorr=acf,
        nrl_autocorr=nrl_ac,
    )


def nucleosome_phasing(
    tw: TrackWindows,
    *,
    groups: Sequence[Any] | None = None,
    smooth_bp: int | None = 15,
    min_distance_bp: float = 120.0,
    prominence: float | None = None,
    min_lag_bp: float = 120.0,
    max_lag_bp: float = 1000.0,
) -> dict[Any, PhasingResult]:
    """Per-group nucleosome-positioning metrics from a dyad :class:`TrackWindows`.

    Collapses regions (optionally by ``groups``) to a mean dyad profile, then detects
    the phased dyad peaks (+1/-1 nucleosome offsets, NDR width) and estimates the
    nucleosome repeat length from both peak spacing and profile autocorrelation.

    Returns ``{group: PhasingResult}`` (group ``'all'`` when ``groups`` is None).
    """
    if groups is None:
        groups = np.array(["all"] * tw.n_regions)
    groups = np.asarray(groups)
    out: dict[Any, PhasingResult] = {}
    for g in pd.unique(groups):
        prof = tw.subset(groups == g).mean_profile()
        prof = _smooth(prof, smooth_bp)
        out[g] = _phasing_from_profile(
            tw.positions,
            prof,
            min_distance_bp=min_distance_bp,
            prominence=prominence,
            min_lag_bp=min_lag_bp,
            max_lag_bp=max_lag_bp,
        )
    return out


def plot_dyad_phasing(
    tw: TrackWindows,
    *,
    groups: Sequence[Any] | None = None,
    smooth_bp: int | None = 15,
    min_distance_bp: float = 120.0,
    ax=None,
    palette: dict | None = None,
    mark_peaks: bool = True,
    ylabel: str | None = None,
    title: str | None = None,
):
    """Fine dyad metaprofile with detected nucleosome peaks marked per group.

    Returns ``(ax, {group: PhasingResult})``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.4))
    phas = nucleosome_phasing(
        tw, groups=groups, smooth_bp=smooth_bp, min_distance_bp=min_distance_bp
    )
    for g, pr in phas.items():
        color = None if palette is None else palette.get(g)
        (line,) = ax.plot(pr.positions, pr.profile, label=str(g), color=color)
        if mark_peaks and pr.peak_positions.size:
            yv = np.interp(pr.peak_positions, pr.positions, pr.profile)
            ax.scatter(
                pr.peak_positions,
                yv,
                s=22,
                color=line.get_color(),
                zorder=5,
                edgecolor="k",
                linewidth=0.3,
            )
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("position relative to site center (bp)")
    ax.set_ylabel(ylabel or f"{tw.track_name} (dyad density)")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    return ax, phas


def overlay_track_with_reads(
    tw: TrackWindows,
    data_matrix: np.ndarray,
    *,
    val_matrix: np.ndarray | None = None,
    read_groups: Sequence[Any] | None = None,
    site_groups: Sequence[Any] | None = None,
    read_positions: np.ndarray | None = None,
    mod_label: str = "fraction methylated",
    track_label: str | None = None,
    smooth_bp: int | None = 20,
    palette: dict | None = None,
    title: str | None = None,
):
    """Headline overlay: track metaprofile stacked above the read methylation pileup.

    Two vertically-stacked panels share an x-axis of *position relative to site
    center*:

    * **top** -- the continuous track metaprofile(s), grouped by ``site_groups``
      (one line per group of *regions*).
    * **bottom** -- the single-read methylation pileup(s), grouped by ``read_groups``
      (one line per group of *reads*; e.g. bound vs unbound).

    ``read_positions`` defaults to ``tw.positions`` and must match
    ``data_matrix.shape[1]``; pass it explicitly if the read window used a different
    span so we can assert alignment.

    Returns the matplotlib ``Figure``.
    """
    import matplotlib.pyplot as plt

    if read_positions is None:
        read_positions = tw.positions
    if data_matrix.shape[1] != len(read_positions):
        raise ValueError(
            f"read window span ({data_matrix.shape[1]}) != positions ({len(read_positions)}); "
            "pass read_positions matching the extraction window_size."
        )

    fig, (ax_t, ax_m) = plt.subplots(
        2,
        1,
        figsize=(7, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
    )
    plot_track_metaprofile(
        tw,
        groups=site_groups,
        smooth_bp=smooth_bp,
        ax=ax_t,
        palette=palette,
        ylabel=track_label or tw.track_name,
    )
    ax_t.set_xlabel("")

    # Bottom: read pileup(s).
    if read_groups is None:
        read_groups = np.array(["reads"] * data_matrix.shape[0])
    read_groups = np.asarray(read_groups)
    for g in pd.unique(read_groups):
        prof = read_pileup(data_matrix, val_matrix, rows=np.where(read_groups == g)[0])
        prof = _smooth(prof, smooth_bp)
        color = None if palette is None else palette.get(g)
        ax_m.plot(
            read_positions,
            prof,
            label=f"{g} (n={int(np.sum(read_groups == g))})",
            color=color,
        )
    ax_m.axvline(0, color="k", lw=0.8, ls=":")
    ax_m.set_xlabel("position relative to site center (bp)")
    ax_m.set_ylabel(mod_label)
    ax_m.legend(frameon=False, fontsize=8)
    if title:
        fig.suptitle(title, y=0.98)
    fig.tight_layout()
    return fig
