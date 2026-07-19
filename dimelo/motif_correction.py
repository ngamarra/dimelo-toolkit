"""Sequence-context (k-mer) deposition/callability bias diagnostic and correction for DiMeLo.

DiMeLo/SpyMeLo methylation calls carry a *multiplicative* per-context bias: the rate at which a
given motif position gets deposited-on and then correctly basecalled depends on its local
sequence (the k-mer around the modified base). This is distinct from the beta-binomial
negative-control null in :mod:`dimelo.background` (which removes biological background
*occupancy*): here we remove *sequence-context* bias, the way flat-fielding removes a detector's
per-pixel gain.

The correction is the multiplicative analog of spectral unmixing: estimate a per-k-mer
efficiency ``e(kmer)`` from a region set with ~flat true occupancy — distal/background sites, or
a naked-DNA / free-enzyme control — then divide it out of the observed signal.

    aggregate    : ``frac_corrected(pos) = frac_obs(pos) * mean(e) / e_local(pos)``
    single read  : ``enrichment = modified_count / sum_over_valid_sites(e(context))``

The k-mer is taken from the reference, centered on the modified base, on the forward strand, and
applied identically during estimation and correction, so the convention cancels in the ratio.

Typical use::

    eff  = estimate_motif_efficiency(bg_h5, ["A,0", "GCH,1"], bg_regions, ref_genome)
    diag = motif_bias_diagnostic(target_h5, "A,0", target_regions, ref_genome, eff)
    prof = correct_aggregate_profile(target_h5, "A,0", target_regions, ref_genome, eff)
    reads = correct_single_reads(target_h5, "A,0", target_regions, ref_genome, eff)

``estimate_motif_efficiency`` / the ``*_profile`` / ``correct_*`` functions do the I/O; the pure
math (``MotifEfficiency`` rates, :func:`flatfield_profile`, :func:`assess_motif_bias`) is
importable and unit-testable without any files.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import load_processed

_VALID_BASES = frozenset("ACGT")


# --------------------------------------------------------------------------------------------
# efficiency model
# --------------------------------------------------------------------------------------------
@dataclass
class MotifEfficiency:
    """Per-k-mer deposition x callability efficiency, estimated from a flat-occupancy control.

    Attributes:
        k: k-mer length (odd; the modified base sits at the center).
        counts: ``{motif: {kmer: [modified, valid]}}`` raw tallies from the control regions.
        min_count: minimum valid calls for a k-mer to use its own rate; below this the motif's
            pooled global rate is used (shrinkage to the mean, avoids noisy rare-context rates).
    """

    k: int
    counts: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    min_count: int = 40

    # -- construction helpers -----------------------------------------------------------------
    def _add(self, motif: str, kmer: str, modified: bool) -> None:
        table = self.counts.setdefault(motif, {})
        cell = table.get(kmer)
        if cell is None:
            cell = [0, 0]
            table[kmer] = cell
        cell[1] += 1
        if modified:
            cell[0] += 1

    # -- queries ------------------------------------------------------------------------------
    @property
    def motifs(self) -> list[str]:
        return list(self.counts)

    def global_rate(self, motif: str) -> float:
        """Pooled methylation rate across all contexts of ``motif`` (the shrinkage target)."""
        table = self.counts.get(motif, {})
        modified = sum(cell[0] for cell in table.values())
        valid = sum(cell[1] for cell in table.values())
        return modified / valid if valid else float("nan")

    def efficiency(self, motif: str, kmer: str | None) -> float:
        """Efficiency for one k-mer, falling back to the motif global rate below ``min_count``.

        ``None`` / unseen / short k-mers return the global rate.
        """
        gm = self.global_rate(motif)
        if kmer is None:
            return gm
        cell = self.counts.get(motif, {}).get(kmer)
        if cell is None or cell[1] < self.min_count:
            return gm
        return cell[0] / cell[1]

    def rate_table(self, motif: str) -> pd.DataFrame:
        """Per-k-mer ``kmer, modified, valid, rate, used`` table (``used`` = passed min_count)."""
        table = self.counts.get(motif, {})
        gm = self.global_rate(motif)
        rows = []
        for kmer, (modified, valid) in sorted(table.items()):
            used = valid >= self.min_count
            rows.append(
                {
                    "kmer": kmer,
                    "modified": modified,
                    "valid": valid,
                    "rate": (modified / valid) if valid else float("nan"),
                    "efficiency": (modified / valid) if used else gm,
                    "used": used,
                }
            )
        return pd.DataFrame(rows, columns=["kmer", "modified", "valid", "rate", "efficiency", "used"])

    def summary(self) -> pd.DataFrame:
        """One row per motif: ``n_contexts, n_used, global_rate, cv`` (coverage-weighted CV of e)."""
        rows = []
        for motif in self.motifs:
            tbl = self.rate_table(motif)
            weights = tbl["valid"].to_numpy(dtype=float)
            eff = tbl["efficiency"].to_numpy(dtype=float)
            if weights.sum() > 0:
                mean = np.average(eff, weights=weights)
                var = np.average((eff - mean) ** 2, weights=weights)
                cv = float(np.sqrt(var) / mean) if mean > 0 else float("nan")
            else:
                cv = float("nan")
            rows.append(
                {
                    "motif": motif,
                    "n_contexts": int(len(tbl)),
                    "n_used": int(tbl["used"].sum()),
                    "global_rate": self.global_rate(motif),
                    "cv": cv,
                }
            )
        return pd.DataFrame(rows, columns=["motif", "n_contexts", "n_used", "global_rate", "cv"])

    # -- persistence --------------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Serialize to JSON (portable, human-inspectable)."""
        payload = {"k": self.k, "min_count": self.min_count, "counts": self.counts}
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> MotifEfficiency:
        payload = json.loads(Path(path).read_text())
        return cls(k=int(payload["k"]), min_count=int(payload["min_count"]), counts=payload["counts"])


# --------------------------------------------------------------------------------------------
# reference-context extraction
# --------------------------------------------------------------------------------------------
def _field_index(dataset_names: Sequence[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(dataset_names)}


def _kmer(seq: str, seq_start: int, genomic_pos: int, half: int) -> str | None:
    """Forward-reference k-mer of length ``2*half+1`` centered on ``genomic_pos``."""
    i = genomic_pos - seq_start
    if i - half < 0 or i + half + 1 > len(seq):
        return None
    kmer = seq[i - half : i + half + 1]
    return kmer if _VALID_BASES.issuperset(kmer) else None


def _iter_read_sites(
    reads: Iterable[tuple],
    ix: dict[str, int],
    fasta,
    half: int,
    thresh: float | None,
    regions_5to3prime: bool,
) -> Iterator[tuple]:
    """Yield ``(read_id, motif, kmer, rel_pos, is_valid, is_mod)`` for every valid motif site.

    ``rel_pos`` is signed distance from the region center, oriented 5'->3' when
    ``regions_5to3prime`` and the region is on the minus strand. ``read_id`` is the row index.
    """
    seq_cache: dict[tuple, tuple[str, int]] = {}
    for read_id, read in enumerate(reads):
        chrom = read[ix["chromosome"]]
        read_start = int(read[ix["read_start"]])
        region_start = int(read[ix["region_start"]])
        region_end = int(read[ix["region_end"]])
        strand = read[ix["region_strand"]]
        motif = read[ix["motif"]]
        mod_vector = np.asarray(read[ix["mod_vector"]])
        val_vector = np.asarray(read[ix["val_vector"]])
        center = (region_start + region_end) // 2

        key = (chrom, region_start, region_end)
        cached = seq_cache.get(key)
        if cached is None:
            seq_start = max(0, region_start - half)
            seq = fasta.fetch(chrom, seq_start, region_end + half).upper()
            cached = (seq, seq_start)
            seq_cache[key] = cached
        seq, seq_start = cached

        valid_idx = np.flatnonzero(val_vector > 0)
        mod_mask = (mod_vector > 0) if thresh is None else (mod_vector >= thresh)
        for i in valid_idx:
            g = read_start + int(i)
            rel = g - center
            if regions_5to3prime and strand == "-":
                rel = -rel
            yield (read_id, motif, _kmer(seq, seq_start, g, half), int(rel), True, bool(mod_mask[i]))


def _open_fasta(ref_genome: str | Path):
    import pysam

    return pysam.FastaFile(str(ref_genome))


# --------------------------------------------------------------------------------------------
# estimation
# --------------------------------------------------------------------------------------------
def estimate_motif_efficiency(
    file: str | Path,
    motifs: list[str] | str,
    regions: str | Path | list[str | Path],
    ref_genome: str | Path,
    *,
    k: int = 5,
    thresh: float | None = None,
    min_count: int = 40,
    window_size: int | None = None,
    **loader_kwargs,
) -> MotifEfficiency:
    """Estimate per-k-mer efficiency ``e(kmer)`` from a flat-occupancy control region set.

    ``regions`` should be background/distal windows (or a naked-DNA / free-enzyme control) where
    true occupancy is ~uniform, so the per-context methylation rate reflects deposition x
    callability alone. Reads are loaded with :func:`load_processed.read_vectors_from_hdf5`.

    Args:
        file: single-read ``.h5`` (from ``parse_bam.extract``) covering ``regions``.
        motifs: motif spec(s), e.g. ``"A,0"`` or ``["A,0", "GCH,1"]``.
        regions: background region .bed path(s) / region strings.
        ref_genome: reference FASTA (indexed) the reads are aligned to.
        k: odd k-mer length centered on the modified base.
        thresh: mod-probability call threshold (0-1). ``None`` treats a pre-called h5's nonzero
            mod entries as methylated.
        min_count: min valid calls before a k-mer uses its own rate (else the global rate).
        window_size: optional centered-window size forwarded to the loader.

    Returns:
        A fitted :class:`MotifEfficiency`.
    """
    if k % 2 == 0:
        raise ValueError("k must be odd so the modified base is centered.")
    motif_list = [motifs] if isinstance(motifs, str) else list(motifs)
    half = k // 2
    data, dataset_names, _ = load_processed.read_vectors_from_hdf5(
        file=str(file), motifs=motif_list, regions=regions, window_size=window_size, **loader_kwargs
    )
    ix = _field_index(dataset_names)
    eff = MotifEfficiency(k=k, min_count=min_count)
    fasta = _open_fasta(ref_genome)
    try:
        for _read_id, motif, kmer, _rel, _valid, is_mod in _iter_read_sites(
            data, ix, fasta, half, thresh, regions_5to3prime=False
        ):
            if kmer is not None:
                eff._add(motif, kmer, is_mod)
    finally:
        fasta.close()
    return eff


# --------------------------------------------------------------------------------------------
# aggregate profiles
# --------------------------------------------------------------------------------------------
def motif_profiles(
    file: str | Path,
    motif: str,
    regions: str | Path | list[str | Path],
    ref_genome: str | Path,
    efficiency: MotifEfficiency,
    *,
    window_size: int = 1000,
    bin_size: int = 25,
    thresh: float | None = None,
    regions_5to3prime: bool = True,
    **loader_kwargs,
) -> pd.DataFrame:
    """Aggregate observed vs motif-expected methylation fraction per position bin.

    ``window_size`` is a RADIUS (matching the rest of dimelo). For each bin the observed
    fraction is ``modified/valid`` and the expected fraction is the coverage-weighted mean
    ``e(context)`` over the same valid sites — i.e. what the metaprofile would look like from
    sequence-context bias alone.

    Returns a DataFrame indexed by bin with columns
    ``position, observed, expected, valid`` (``position`` = bin center bp from region center).
    """
    half = efficiency.k // 2
    n_bins = 2 * window_size // bin_size
    modified = np.zeros(n_bins)
    valid = np.zeros(n_bins)
    expected = np.zeros(n_bins)
    global_rate = efficiency.global_rate(motif)

    data, dataset_names, _ = load_processed.read_vectors_from_hdf5(
        file=str(file), motifs=[motif], regions=regions, window_size=2 * window_size, **loader_kwargs
    )
    ix = _field_index(dataset_names)
    fasta = _open_fasta(ref_genome)
    try:
        for _read_id, site_motif, kmer, rel, _valid, is_mod in _iter_read_sites(
            data, ix, fasta, half, thresh, regions_5to3prime
        ):
            if site_motif != motif or not (-window_size <= rel < window_size):
                continue
            b = (rel + window_size) // bin_size
            valid[b] += 1
            expected[b] += efficiency.efficiency(motif, kmer) if kmer is not None else global_rate
            if is_mod:
                modified[b] += 1
    finally:
        fasta.close()

    obs = np.divide(modified, valid, out=np.zeros(n_bins), where=valid > 0)
    exp = np.divide(expected, valid, out=np.full(n_bins, global_rate), where=valid > 0)
    positions = (np.arange(n_bins) * bin_size) - window_size + bin_size / 2.0
    return pd.DataFrame({"position": positions, "observed": obs, "expected": exp, "valid": valid})


def flatfield_profile(
    observed: np.ndarray, expected: np.ndarray, *, mode: str = "ratio"
) -> np.ndarray:
    """Divide the motif-expected efficiency out of an observed profile (pure math).

    ``ratio``  : ``observed * mean(expected) / expected`` — preserves overall level, matches the
                 validated SpyMeLo flat-field.
    ``logit``  : correct on the logit scale (additive there), ``sigmoid(logit(obs) - logit(exp)
                 + logit(mean_obs))`` — better when rates approach 0/1.
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if observed.shape != expected.shape:
        raise ValueError("observed and expected must have the same shape.")
    if mode == "ratio":
        exp_mean = np.nanmean(expected)
        safe = np.where(expected > 0, expected, exp_mean)
        return observed * (exp_mean / safe)
    if mode == "logit":
        def _logit(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return np.log(p / (1 - p))
        obs_mean = np.clip(np.nanmean(observed), 1e-6, 1 - 1e-6)
        z = _logit(observed) - _logit(expected) + np.log(obs_mean / (1 - obs_mean))
        return 1.0 / (1.0 + np.exp(-z))
    raise ValueError("mode must be 'ratio' or 'logit'.")


def correct_aggregate_profile(
    file: str | Path,
    motif: str,
    regions: str | Path | list[str | Path],
    ref_genome: str | Path,
    efficiency: MotifEfficiency,
    *,
    mode: str = "ratio",
    **kwargs,
) -> pd.DataFrame:
    """Convenience: :func:`motif_profiles` + :func:`flatfield_profile`.

    Returns the ``motif_profiles`` frame plus a ``corrected`` column.
    """
    frame = motif_profiles(file, motif, regions, ref_genome, efficiency, **kwargs)
    frame["corrected"] = flatfield_profile(
        frame["observed"].to_numpy(), frame["expected"].to_numpy(), mode=mode
    )
    return frame


# --------------------------------------------------------------------------------------------
# single-read correction
# --------------------------------------------------------------------------------------------
def correct_single_reads(
    file: str | Path,
    motif: str,
    regions: str | Path | list[str | Path],
    ref_genome: str | Path,
    efficiency: MotifEfficiency,
    *,
    window_size: int | None = None,
    thresh: float | None = None,
    regions_5to3prime: bool = True,
    return_null_vectors: bool = False,
    **loader_kwargs,
):
    """Per-read context-adjusted occupancy (the single-molecule analog of the flat-field).

    For each read, over its valid motif sites: ``modified_count`` (k), ``valid_count`` (n) and
    ``expected_count`` (``sum e(context)``). The context-adjusted enrichment ``k / expected_count``
    (>1 means more methylation than sequence context predicts) and the logit residual give a
    per-molecule signal with sequence bias removed — usable for clustering / co-occupancy /
    footprinting instead of the raw fraction.

    Returns a DataFrame (one row per read): ``read_id, chromosome, modified_count, valid_count,
    observed_fraction, expected_fraction, expected_count, enrichment, logit_residual``. If
    ``return_null_vectors`` also returns ``{read_id: (rel_positions, expected_probs, is_mod)}``
    giving the per-site context null for GLMM/footprint conditioning.
    """
    half = efficiency.k // 2
    global_rate = efficiency.global_rate(motif)
    data, dataset_names, _ = load_processed.read_vectors_from_hdf5(
        file=str(file), motifs=[motif], regions=regions, window_size=window_size, **loader_kwargs
    )
    ix = _field_index(dataset_names)

    agg: dict[int, list[float]] = {}          # read_id -> [k, n, expected_k]
    nulls: dict[int, list[list]] = {}
    fasta = _open_fasta(ref_genome)
    try:
        for read_id, site_motif, kmer, rel, _valid, is_mod in _iter_read_sites(
            data, ix, fasta, half, thresh, regions_5to3prime
        ):
            if site_motif != motif:
                continue
            e = efficiency.efficiency(motif, kmer) if kmer is not None else global_rate
            row = agg.setdefault(read_id, [0.0, 0.0, 0.0])
            row[0] += 1.0 if is_mod else 0.0
            row[1] += 1.0
            row[2] += e
            if return_null_vectors:
                nz = nulls.setdefault(read_id, [[], [], []])
                nz[0].append(rel)
                nz[1].append(e)
                nz[2].append(1 if is_mod else 0)
    finally:
        fasta.close()

    chroms = {rid: data[rid][ix["chromosome"]] for rid in agg}
    rows = []
    for read_id, (k, n, exp_k) in agg.items():
        obs_frac = k / n if n else float("nan")
        exp_frac = exp_k / n if n else float("nan")
        enrichment = k / exp_k if exp_k > 0 else float("nan")
        of = min(max(obs_frac, 1e-6), 1 - 1e-6)
        ef = min(max(exp_frac, 1e-6), 1 - 1e-6)
        logit_resid = np.log(of / (1 - of)) - np.log(ef / (1 - ef))
        rows.append(
            {
                "read_id": read_id,
                "chromosome": chroms[read_id],
                "modified_count": int(k),
                "valid_count": int(n),
                "observed_fraction": obs_frac,
                "expected_fraction": exp_frac,
                "expected_count": exp_k,
                "enrichment": enrichment,
                "logit_residual": float(logit_resid),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "read_id", "chromosome", "modified_count", "valid_count", "observed_fraction",
            "expected_fraction", "expected_count", "enrichment", "logit_residual",
        ],
    ).sort_values("read_id").reset_index(drop=True)
    if return_null_vectors:
        vectors = {
            rid: (np.array(v[0]), np.array(v[1], dtype=float), np.array(v[2], dtype=int))
            for rid, v in nulls.items()
        }
        return frame, vectors
    return frame


# --------------------------------------------------------------------------------------------
# diagnostic
# --------------------------------------------------------------------------------------------
@dataclass
class MotifBiasDiagnostic:
    """Result of :func:`motif_bias_diagnostic`.

    Attributes:
        motif: the motif assessed.
        cv: coverage-weighted CV of the motif-efficiency track across the window (amplitude of
            context variation; ~0 means context is flat, nothing to correct).
        r2: fraction of the observed-profile variance explained by the motif-expected profile
            (how much of the pattern *could* be sequence bias).
        pearson: correlation of observed vs motif-expected profiles.
        delta_self: change in observed's own shape after correction (L2, normalized) — how much
            correcting actually moves the profile.
        is_confounded: verdict (cv or r2 above threshold).
        profiles: the underlying ``position, observed, expected, corrected`` frame.
    """

    motif: str
    cv: float
    r2: float
    pearson: float
    delta_self: float
    is_confounded: bool
    profiles: pd.DataFrame

    def to_dict(self) -> dict[str, float | bool | str]:
        return {
            "motif": self.motif,
            "cv": self.cv,
            "r2": self.r2,
            "pearson": self.pearson,
            "delta_self": self.delta_self,
            "is_confounded": self.is_confounded,
        }


def assess_motif_bias(
    observed: np.ndarray,
    expected: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    motif: str = "",
    cv_threshold: float = 0.05,
    r2_threshold: float = 0.10,
    mode: str = "ratio",
) -> MotifBiasDiagnostic:
    """Score whether a motif-expected track confounds an observed profile (pure math).

    ``cv`` measures context-track amplitude; ``r2`` how much of the observed pattern the motif
    track explains; ``delta_self`` how much correction moves the observed profile. Flags
    confounding when ``cv > cv_threshold`` or ``r2 > r2_threshold``.
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    weights = np.ones_like(expected) if valid is None else np.asarray(valid, dtype=float)
    if weights.sum() > 0:
        mean = np.average(expected, weights=weights)
        var = np.average((expected - mean) ** 2, weights=weights)
        cv = float(np.sqrt(var) / mean) if mean > 0 else float("nan")
    else:
        cv = float("nan")

    o = observed - observed.mean()
    e = expected - expected.mean()
    denom = np.sqrt((o**2).sum() * (e**2).sum())
    pearson = float((o * e).sum() / denom) if denom > 0 else 0.0
    r2 = pearson**2

    corrected = flatfield_profile(observed, expected, mode=mode)
    c = corrected - corrected.mean()
    scale = np.sqrt((o**2).sum()) or 1.0
    delta_self = float(np.sqrt(((c - o) ** 2).sum()) / scale)

    profiles = pd.DataFrame(
        {"observed": observed, "expected": expected, "corrected": corrected}
    )
    is_confounded = bool((np.isfinite(cv) and cv > cv_threshold) or r2 > r2_threshold)
    return MotifBiasDiagnostic(
        motif=motif, cv=cv, r2=r2, pearson=pearson, delta_self=delta_self,
        is_confounded=is_confounded, profiles=profiles,
    )


def motif_bias_diagnostic(
    file: str | Path,
    motif: str,
    regions: str | Path | list[str | Path],
    ref_genome: str | Path,
    efficiency: MotifEfficiency,
    *,
    cv_threshold: float = 0.05,
    r2_threshold: float = 0.10,
    mode: str = "ratio",
    **kwargs,
) -> MotifBiasDiagnostic:
    """Build observed & motif-expected profiles at ``regions`` and score the motif confound.

    Wraps :func:`motif_profiles` + :func:`assess_motif_bias`; the returned
    :class:`MotifBiasDiagnostic` carries the ``position``-indexed profiles for plotting.
    """
    frame = motif_profiles(file, motif, regions, ref_genome, efficiency, **kwargs)
    diag = assess_motif_bias(
        frame["observed"].to_numpy(), frame["expected"].to_numpy(), frame["valid"].to_numpy(),
        motif=motif, cv_threshold=cv_threshold, r2_threshold=r2_threshold, mode=mode,
    )
    diag.profiles.insert(0, "position", frame["position"].to_numpy())
    return diag


def plot_motif_correction(diagnostic: MotifBiasDiagnostic, ax=None):
    """Overlay observed / motif-expected / corrected profiles with the diagnostic verdict.

    Returns the matplotlib Axes. Matplotlib is imported lazily.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 4))
    df = diagnostic.profiles
    x = df["position"].to_numpy() if "position" in df else np.arange(len(df))
    ax.plot(x, df["observed"], color="0.3", lw=1.6, label="observed")
    ax.plot(x, df["expected"], color="#d62728", lw=1.4, ls="--", label="motif-expected e(kmer)")
    ax.plot(x, df["corrected"], color="#1f77b4", lw=2.0, label="corrected")
    ax.axvline(0, ls=":", color="grey")
    verdict = "CONFOUNDED" if diagnostic.is_confounded else "motif not a confound"
    ax.set_title(
        f"{diagnostic.motif}: CV(e)={diagnostic.cv:.3f}  R²(obs~motif)={diagnostic.r2:.3f}  "
        f"Δcorr={diagnostic.delta_self:.3f}\n{verdict}"
    )
    ax.set_xlabel("position (bp from region center)")
    ax.set_ylabel("fraction modified")
    ax.legend(fontsize=8)
    return ax
