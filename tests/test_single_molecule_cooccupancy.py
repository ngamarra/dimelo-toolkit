"""Tests for dimelo.single_molecule_cooccupancy: stratified odds ratios, permutation null,
detection-sensitivity simulation, presence calling, and the per-read loader.

Pure-math tests (odds ratios, permutation, sensitivity) use plain arrays. One integration test
exercises the h5 loader + full analysis against the checked-in CTCF demo extract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dimelo import single_molecule_cooccupancy as smc

_REPO = Path(__file__).resolve().parents[1]
_EXTRACT = _REPO / "dimelo" / "test" / "output" / "ctcf_demo_extract"
_H5 = _EXTRACT / "reads.combined_basemods.h5"
_BED = _EXTRACT / "regions.processed.bed"


# ---------------------------------------------------------------- odds ratios
def test_pooled_odds_ratio_known_table():
    assert smc.pooled_odds_ratio([(8, 2, 2, 8)]) == pytest.approx((8 * 8) / (2 * 2))


def test_mantel_haenszel_recovers_common_or():
    # two identical strata each with OR = 4 -> MH OR = 4
    res = smc.mantel_haenszel([(4, 2, 1, 2), (4, 2, 1, 2)])
    assert res.odds_ratio == pytest.approx(4.0, rel=1e-6)
    assert res.n_informative == 2


def test_mantel_haenszel_simpson_paradox():
    # each stratum is independent (OR=1) but pooled looks associated
    strata = [(25, 25, 25, 25), (1, 9, 9, 81)]
    assert smc.pooled_odds_ratio(strata) > 2.0          # Simpson-inflated
    assert smc.mantel_haenszel(strata).odds_ratio == pytest.approx(1.0, abs=0.02)


def test_mantel_haenszel_independence_not_significant():
    res = smc.mantel_haenszel([(25, 25, 25, 25)] * 4)
    assert res.odds_ratio == pytest.approx(1.0, abs=1e-6)
    assert res.p_value > 0.5


def test_mantel_haenszel_empty_returns_nan():
    res = smc.mantel_haenszel([(1, 0, 0, 0)])   # single-margin stratum -> no information
    assert np.isnan(res.odds_ratio)


def test_contingency_strata_grouping():
    loci = ["g1", "g1", "g2"]
    cells, arrays = smc.contingency_strata(loci, [True, False, True], [True, True, False])
    assert len(cells) == 2 and len(arrays) == 2
    # g1: (E+O+, E+O-, E-O+, E-O-) = (1,0,1,0)
    assert cells[0] == (1, 0, 1, 0)


# ---------------------------------------------------------------- permutation null
def test_permutation_null_centered_near_one_under_independence():
    rng = np.random.default_rng(1)
    arrays = []
    for _ in range(300):
        n = 4
        arrays.append((rng.random(n) < 0.5, rng.random(n) < 0.5))
    null = smc.permutation_null_or(arrays, n_perm=50, seed=0)
    assert null.size > 0
    assert 0.7 < float(np.median(null)) < 1.4


# ---------------------------------------------------------------- detection sensitivity
def _outcome_arrays(n_loci=400, size=5, rate=0.3, seed=3):
    rng = np.random.default_rng(seed)
    return [rng.random(size) < rate for _ in range(n_loci)]


def test_detection_sensitivity_identity_no_loss():
    arr = _outcome_arrays()
    obs = smc.detection_sensitivity_or(arr, target_exposure_rate=0.15, true_or=2.5,
                                       detection=1.0, suppression=0.0, reps=20, seed=0)
    assert 1.9 < obs < 3.3            # observed ~= true when nothing is lost


def test_detection_sensitivity_unbiased_under_null_when_undersampled():
    arr = _outcome_arrays()
    obs = smc.detection_sensitivity_or(arr, target_exposure_rate=0.10, true_or=1.0,
                                       detection=0.5, suppression=0.0, reps=20, seed=0)
    assert 0.75 < obs < 1.35          # uniform undersampling leaves OR=1 unbiased


def test_conditional_suppression_biases_or_downward():
    arr = _outcome_arrays()
    base = smc.detection_sensitivity_or(arr, 0.15, true_or=2.0, detection=1.0,
                                        suppression=0.0, reps=20, seed=0)
    supp = smc.detection_sensitivity_or(arr, 0.15, true_or=2.0, detection=1.0,
                                        suppression=0.4, reps=20, seed=0)
    assert supp < base                # double-positive suppression pulls the OR toward/below 1


# ---------------------------------------------------------------- presence threshold
def test_presence_threshold_quantile():
    vals = np.linspace(0, 1, 101)
    assert smc.presence_threshold(vals, fpr=0.05) == pytest.approx(0.95, abs=1e-6)


# ---------------------------------------------------------------- integration
@pytest.mark.skipif(not (_H5.exists() and _BED.exists()), reason="CTCF demo extract not available")
def test_integration_loader_and_conditional_cooccupancy():
    frame = smc.per_read_window_fractions(_H5, ["A,0", "CG,0"], _BED, window=1000)
    assert {"locus", "A,0_fraction", "CG,0_fraction"}.issubset(frame.columns)
    assert frame["A,0_fraction"].between(0, 1).all()
    if len(frame) >= 5:
        thA = smc.presence_threshold(frame["A,0_fraction"], fpr=0.5)
        thC = smc.presence_threshold(frame["CG,0_fraction"], fpr=0.5)
        res = smc.conditional_cooccupancy(
            frame, frame["A,0_fraction"] > thA, frame["CG,0_fraction"] > thC,
            permute=True, n_perm=20,
        )
        assert isinstance(res.mantel_haenszel, smc.OddsRatioResult)
        assert res.n_reads == len(frame)
