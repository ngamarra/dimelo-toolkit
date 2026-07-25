"""Conditional single-molecule co-occupancy of two modification channels at anchored regions.

DiMeLo/SpyMeLo can read two factors on the *same* molecule (e.g. Pol II via 6mA and H3K4me3
via Gp5mC). The naive question "do reads carrying mark A also carry mark B?" is answered by a
pooled 2x2 odds ratio — but that is confounded by *region activity* (active promoters carry more
of both), i.e. Simpson's paradox. The cis question is whether the two marks are coupled *within*
a region, controlling for its identity.

This module provides the region-stratified machinery, validated against the factorial SpyMeLo
controls:

- :func:`mantel_haenszel` — common odds ratio across per-region 2x2 strata (Robins-Breslow-
  Greenland CI + Mantel-Haenszel test); the Simpson-safe estimator. Compare to
  :func:`pooled_odds_ratio` to see how much of an apparent association is between-region activity.
- :func:`conditional_logistic_or` — regression analog (locus-stratified), an independent check.
- :func:`permutation_null_or` — within-region label shuffle (preserves both margins, destroys cis
  pairing) for an empirical null.
- :func:`detection_sensitivity_or` / :func:`infer_true_or` — how a true within-region OR maps to
  the observed one under (i) uniform detection loss ``d`` (e.g. a heterozygous tag under-sampling
  one channel) and (ii) conditional suppression ``s`` of double-positives (e.g. suppressive
  basecalling crosstalk). Both only *remove* detections, so they cannot manufacture co-occupancy.
- :func:`presence_threshold` / :func:`per_read_window_fractions` — call per-read presence against a
  negative-control distribution and pull per-read window fractions straight from a dimelo ``.h5``.

Pure-math functions (odds ratios, permutation, sensitivity) take plain arrays and are unit-
testable without any files; the loader bridges from :func:`dimelo.load_processed.read_vectors_from_hdf5`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import load_processed


# --------------------------------------------------------------------------------------------
# odds ratios over 2x2 strata
# --------------------------------------------------------------------------------------------
@dataclass
class OddsRatioResult:
    """Common odds ratio across strata with confidence interval and test.

    Attributes:
        odds_ratio: Mantel-Haenszel common OR (``nan`` if no informative stratum).
        ci_low, ci_high: 95% CI from the Robins-Breslow-Greenland ln-OR variance.
        chi2, p_value: Mantel-Haenszel (continuity-corrected) test of OR=1.
        n_strata: strata supplied; n_informative: strata with >=2 reads used.
    """

    odds_ratio: float
    ci_low: float
    ci_high: float
    chi2: float
    p_value: float
    n_strata: int
    n_informative: int


Stratum = tuple[int, int, int, int]   # (a=E+O+, b=E+O-, c=E-O+, d=E-O-)


def pooled_odds_ratio(strata: Sequence[Stratum]) -> float:
    """Odds ratio of the single pooled 2x2 table (ignores strata; Simpson-prone)."""
    a = sum(s[0] for s in strata)
    b = sum(s[1] for s in strata)
    c = sum(s[2] for s in strata)
    d = sum(s[3] for s in strata)
    return (a * d) / (b * c) if b * c > 0 else float("nan")


def mantel_haenszel(strata: Sequence[Stratum]) -> OddsRatioResult:
    """Mantel-Haenszel common OR + RBG 95% CI + MH chi-square over 2x2 strata ``(a,b,c,d)``.

    Strata with fewer than two reads, or with a zero row/column margin, contribute no
    information (correctly) and are skipped in the estimator.
    """
    R = S = PR = PSQR = QS = num_a = exp_a = den_v = 0.0
    n_inf = 0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n < 2:
            continue
        n_inf += 1
        R += a * d / n
        S += b * c / n
        P = (a + d) / n
        Q = (b + c) / n
        PR += P * (a * d / n)
        QS += Q * (b * c / n)
        PSQR += P * (b * c / n) + Q * (a * d / n)
        num_a += a
        exp_a += (a + b) * (a + c) / n
        den_v += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    if R == 0 or S == 0:
        return OddsRatioResult(float("nan"), float("nan"), float("nan"), float("nan"),
                               float("nan"), len(strata), n_inf)
    from scipy.stats import chi2 as _chi2
    odds = R / S
    se = float(np.sqrt(PR / (2 * R * R) + PSQR / (2 * R * S) + QS / (2 * S * S)))
    chi2 = ((abs(num_a - exp_a) - 0.5) ** 2 / den_v) if den_v > 0 else float("nan")
    p = float(_chi2.sf(chi2, 1)) if np.isfinite(chi2) else float("nan")
    return OddsRatioResult(float(odds), float(odds * np.exp(-1.96 * se)),
                           float(odds * np.exp(1.96 * se)), float(chi2), p, len(strata), n_inf)


def contingency_strata(
    loci: Sequence, exposure: Sequence[bool], outcome: Sequence[bool]
) -> tuple[list[Stratum], list[tuple[np.ndarray, np.ndarray]]]:
    """Group aligned per-read ``(exposure, outcome)`` booleans by ``locus`` key.

    Returns the list of 2x2 cells ``(a,b,c,d)`` per locus, and, per locus, the aligned
    ``(exposure, outcome)`` boolean arrays (for :func:`permutation_null_or`).
    """
    by = defaultdict(lambda: [[], []])
    for lo, e, o in zip(loci, exposure, outcome, strict=True):
        by[lo][0].append(bool(e))
        by[lo][1].append(bool(o))
    cells: list[Stratum] = []
    arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for ev, ov in by.values():
        ev = np.asarray(ev)
        ov = np.asarray(ov)
        cells.append((int(np.sum(ev & ov)), int(np.sum(ev & ~ov)),
                      int(np.sum(~ev & ov)), int(np.sum(~ev & ~ov))))
        arrays.append((ev, ov))
    return cells, arrays


def permutation_null_or(
    locus_arrays: Sequence[tuple[np.ndarray, np.ndarray]], *, n_perm: int = 200, seed: int = 0
) -> np.ndarray:
    """Within-locus shuffle null for the MH OR.

    Permutes the *outcome* labels within each locus (preserving both per-locus margins,
    destroying cis pairing) and recomputes the MH OR ``n_perm`` times. Compare the observed OR
    to this distribution for an empirical p-value.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        cells = []
        for ev, ov in locus_arrays:
            os_ = rng.permutation(ov)
            cells.append((int(np.sum(ev & os_)), int(np.sum(ev & ~os_)),
                          int(np.sum(~ev & os_)), int(np.sum(~ev & ~os_))))
        r = mantel_haenszel(cells).odds_ratio
        if np.isfinite(r):
            out.append(r)
    return np.asarray(out)


def conditional_logistic_or(
    loci: Sequence, exposure: Sequence[bool], outcome: Sequence[bool]
) -> dict | None:
    """Locus-stratified conditional logistic regression; ``exp(coef)`` = within-locus OR.

    Independent cross-check of :func:`mantel_haenszel`. Returns ``{odds_ratio, ci_low, ci_high,
    p_value}`` or ``None`` if statsmodels is unavailable or the design is degenerate.
    """
    try:
        from statsmodels.discrete.conditional_models import ConditionalLogit
    except Exception:  # noqa: BLE001
        return None
    gid = {k: i for i, k in enumerate(dict.fromkeys(loci))}
    y = np.asarray([int(o) for o in outcome], dtype=float)
    x = np.asarray([[int(e)] for e in exposure], dtype=float)
    g = np.asarray([gid[k] for k in loci])
    if len(set(g)) < 2 or x.sum() == 0 or x.sum() == len(x):
        return None
    try:
        res = ConditionalLogit(y, x, groups=g).fit(disp=0)
        ci = res.conf_int()[0]
        return {"odds_ratio": float(np.exp(res.params[0])), "ci_low": float(np.exp(ci[0])),
                "ci_high": float(np.exp(ci[1])), "p_value": float(res.pvalues[0])}
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------------------------
# detection sensitivity (undersampling / crosstalk)
# --------------------------------------------------------------------------------------------
def detection_sensitivity_or(
    locus_outcome_arrays: Sequence[np.ndarray],
    target_exposure_rate: float,
    true_or: float,
    *,
    detection: float = 1.0,
    suppression: float = 0.0,
    reps: int = 12,
    seed: int = 0,
) -> float:
    """Mean OBSERVED MH-OR when the TRUE within-locus OR is ``true_or``, under detection loss.

    Simulates the (undersampled) exposure channel against the real, fully-sampled ``outcome``
    per locus: draws true exposure with within-locus log-odds ``ln(true_or)`` on the outcome
    (intercept calibrated so the *detected* exposure rate matches ``target_exposure_rate``),
    applies uniform detection probability ``detection`` (e.g. a heterozygous tag) and, on true
    double-positive reads, conditional ``suppression`` of the exposure call (e.g. suppressive
    basecalling crosstalk). Both only remove detections, so the OR cannot be inflated spuriously.

    With ``detection=1, suppression=0`` the returned observed OR ~= ``true_or`` (identity check).
    """
    rng = np.random.default_rng(seed)
    g_all = np.concatenate([np.asarray(o, dtype=float) for o in locus_outcome_arrays])
    beta = np.log(true_or)
    lo, hi = -30.0, 30.0                       # calibrate intercept for the detected exposure rate
    for _ in range(80):
        mid = (lo + hi) / 2
        p = 1.0 / (1.0 + np.exp(-(mid + beta * g_all)))
        mean_obs = detection * p.mean() - suppression * detection * (g_all * p).mean()
        if mean_obs < target_exposure_rate:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2
    ors = []
    for _ in range(reps):
        cells = []
        for o in locus_outcome_arrays:
            o = np.asarray(o).astype(bool)
            p = 1.0 / (1.0 + np.exp(-(alpha + beta * o)))
            estar = rng.random(len(o)) < p
            edet = estar & (rng.random(len(o)) < detection)
            if suppression > 0:
                dp = edet & o
                edet = edet & ~(dp & (rng.random(len(o)) < suppression))
            cells.append((int(np.sum(edet & o)), int(np.sum(edet & ~o)),
                          int(np.sum(~edet & o)), int(np.sum(~edet & ~o))))
        r = mantel_haenszel(cells).odds_ratio
        if np.isfinite(r):
            ors.append(r)
    return float(np.nanmean(ors)) if ors else float("nan")


def infer_true_or(
    locus_outcome_arrays: Sequence[np.ndarray],
    observed_or: float,
    target_exposure_rate: float,
    *,
    detection: float = 1.0,
    suppression: float = 0.0,
    true_or_grid: Sequence[float] | None = None,
    reps: int = 12,
    seed: int = 0,
) -> float:
    """True within-locus OR consistent with ``observed_or`` under the given detection loss.

    Inverts :func:`detection_sensitivity_or` by interpolation over ``true_or_grid``.
    """
    grid = np.asarray(true_or_grid if true_or_grid is not None
                      else [0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0], dtype=float)
    obs = np.array([detection_sensitivity_or(locus_outcome_arrays, target_exposure_rate, t,
                                             detection=detection, suppression=suppression,
                                             reps=reps, seed=seed) for t in grid])
    order = np.argsort(obs)
    return float(np.interp(observed_or, obs[order], grid[order]))


# --------------------------------------------------------------------------------------------
# presence calling + per-read loader
# --------------------------------------------------------------------------------------------
def presence_threshold(control_values: Sequence[float], *, fpr: float = 0.05) -> float:
    """Presence threshold = the ``1-fpr`` quantile of a negative-control per-read value set.

    Calling a read "present" when its value exceeds this bounds the per-read false-positive rate
    at ``fpr`` against the control (e.g. a prequench/IgG barcode), rather than an arbitrary cutoff.
    """
    return float(np.percentile(np.asarray(control_values, dtype=float), 100 * (1 - fpr)))


def per_read_window_fractions(
    file: str | Path,
    motifs: list[str],
    regions: str | Path | list[str | Path],
    *,
    window: int = 1000,
    window_size: int | None = None,
    thresh: float | None = None,
    regions_5to3prime: bool = True,
    **loader_kwargs,
) -> pd.DataFrame:
    """Per-read modified/valid counts and fraction within +-``window`` of each region center.

    Wraps :func:`dimelo.load_processed.read_vectors_from_hdf5` and reduces each read to, per
    motif, ``<motif>_modified`` / ``<motif>_valid`` / ``<motif>_fraction`` inside the window,
    plus a ``locus`` key ``(chromosome, region_start, region_end, strand)`` for stratification.

    Args:
        window: half-width (bp) of the count window around the region center.
        window_size: optional centered-window size forwarded to the loader.
        thresh: mod-probability call threshold (0-1); ``None`` treats nonzero mod entries in a
            pre-called h5 as methylated.
    """
    data, names, _ = load_processed.read_vectors_from_hdf5(
        file=str(file), motifs=motifs, regions=regions, window_size=window_size, **loader_kwargs
    )
    ix = {n: i for i, n in enumerate(names)}
    rows: dict[tuple, dict] = {}
    for rd in data:
        mv = np.asarray(rd[ix["mod_vector"]])
        vv = np.asarray(rd[ix["val_vector"]])
        rs = int(rd[ix["read_start"]])
        rgs = int(rd[ix["region_start"]])
        rge = int(rd[ix["region_end"]])
        strand = rd[ix["region_strand"]]
        motif = rd[ix["motif"]]
        center = (rgs + rge) // 2
        mp = np.flatnonzero(mv > 0) + rs - center
        vp = np.flatnonzero(vv > 0) + rs - center
        if thresh is not None:
            mp = np.flatnonzero(mv >= thresh) + rs - center
        if regions_5to3prime and strand == "-":
            mp = -mp
            vp = -vp
        nmod = int(np.sum(np.abs(mp) < window))
        nval = int(np.sum(np.abs(vp) < window))
        rkey = (rd[ix["read_name"]], rgs, rge, strand)
        row = rows.setdefault(rkey, {"read_name": rd[ix["read_name"]],
                                     "chromosome": rd[ix["chromosome"]],
                                     "region_start": rgs, "region_end": rge, "strand": strand,
                                     "locus": (rd[ix["chromosome"]], rgs, rge, strand)})
        row[f"{motif}_modified"] = row.get(f"{motif}_modified", 0) + nmod
        row[f"{motif}_valid"] = row.get(f"{motif}_valid", 0) + nval
    frame = pd.DataFrame(list(rows.values()))
    for m in motifs:
        mod = frame.get(f"{m}_modified", pd.Series(0, index=frame.index)).fillna(0)
        val = frame.get(f"{m}_valid", pd.Series(0, index=frame.index)).fillna(0)
        frame[f"{m}_modified"] = mod.astype(int)
        frame[f"{m}_valid"] = val.astype(int)
        frame[f"{m}_fraction"] = np.divide(mod, val, out=np.zeros(len(frame), float), where=val > 0)
    return frame


@dataclass
class CooccupancyResult:
    """Bundled output of :func:`conditional_cooccupancy`."""

    pooled_or: float
    mantel_haenszel: OddsRatioResult
    conditional_logit: dict | None
    permutation_p: float | None
    n_reads: int


def conditional_cooccupancy(
    reads: pd.DataFrame, exposure: Sequence[bool], outcome: Sequence[bool],
    *, locus_col: str = "locus", permute: bool = True, n_perm: int = 200, seed: int = 0,
    conditional_logit: bool = True,
) -> CooccupancyResult:
    """Full conditional co-occupancy from per-read presence calls: pooled vs MH OR + checks.

    ``exposure`` / ``outcome`` are per-read boolean presence calls aligned to ``reads`` (e.g.
    from :func:`presence_threshold` on two channels). Compares the Simpson-prone pooled OR to the
    per-locus Mantel-Haenszel OR, with optional conditional-logit and permutation-null checks.
    """
    loci = list(reads[locus_col])
    exposure = list(exposure)
    outcome = list(outcome)
    cells, arrays = contingency_strata(loci, exposure, outcome)
    mh = mantel_haenszel(cells)
    cl = conditional_logistic_or(loci, exposure, outcome) if conditional_logit else None
    perm_p = None
    if permute:
        null = permutation_null_or(arrays, n_perm=n_perm, seed=seed)
        if len(null) and np.isfinite(mh.odds_ratio):
            perm_p = float((np.sum(np.abs(np.log(null)) >= abs(np.log(mh.odds_ratio))) + 1) / (len(null) + 1))
    return CooccupancyResult(pooled_odds_ratio(cells), mh, cl, perm_p, len(reads))
