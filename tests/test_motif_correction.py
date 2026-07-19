"""Tests for dimelo.motif_correction: k-mer efficiency model, flat-field correction, diagnostic.

The pure-math tests (efficiency rates/fallback/persistence, flatfield_profile, assess_motif_bias)
run with no I/O. One integration test exercises the full h5+FASTA path against the checked-in
CTCF demo extract, skipped when that fixture (or the CHM13v2 FASTA) is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dimelo import motif_correction as mc

_REPO = Path(__file__).resolve().parents[1]
_EXTRACT = _REPO / "dimelo" / "test" / "output" / "ctcf_demo_extract"
_H5 = _EXTRACT / "reads.combined_basemods.h5"
_BED = _EXTRACT / "regions.processed.bed"
_FASTA = _REPO / "dimelo" / "test" / "output" / "chm13v2.0.fasta"


# ------------------------------------------------------------------ MotifEfficiency
def _toy_efficiency() -> mc.MotifEfficiency:
    eff = mc.MotifEfficiency(k=3, min_count=10)
    # AAA: 8/100 = 0.08 (used) ; TAT: 1/2 (below min_count -> falls back to global)
    eff.counts = {"A,0": {"AAA": [8, 100], "TAT": [1, 2]}}
    return eff


def test_efficiency_rate_uses_own_when_above_min_count():
    eff = _toy_efficiency()
    assert eff.efficiency("A,0", "AAA") == pytest.approx(0.08)


def test_efficiency_falls_back_to_global_below_min_count_or_unseen():
    eff = _toy_efficiency()
    global_rate = eff.global_rate("A,0")  # (8+1)/(100+2)
    assert global_rate == pytest.approx(9 / 102)
    assert eff.efficiency("A,0", "TAT") == pytest.approx(global_rate)  # below min_count
    assert eff.efficiency("A,0", "GGG") == pytest.approx(global_rate)  # unseen
    assert eff.efficiency("A,0", None) == pytest.approx(global_rate)   # no context


def test_efficiency_rate_table_and_summary():
    eff = _toy_efficiency()
    table = eff.rate_table("A,0").set_index("kmer")
    assert bool(table.loc["AAA", "used"]) is True
    assert bool(table.loc["TAT", "used"]) is False
    assert table.loc["TAT", "efficiency"] == pytest.approx(eff.global_rate("A,0"))
    summary = eff.summary()
    assert set(summary.columns) == {"motif", "n_contexts", "n_used", "global_rate", "cv"}
    assert int(summary.loc[0, "n_contexts"]) == 2
    assert int(summary.loc[0, "n_used"]) == 1


def test_efficiency_save_load_roundtrip(tmp_path):
    eff = _toy_efficiency()
    path = tmp_path / "eff.json"
    eff.save(path)
    loaded = mc.MotifEfficiency.load(path)
    assert loaded.k == eff.k
    assert loaded.min_count == eff.min_count
    assert loaded.efficiency("A,0", "AAA") == pytest.approx(eff.efficiency("A,0", "AAA"))


# ------------------------------------------------------------------ flatfield_profile
def test_flatfield_ratio_no_bias_is_identity():
    observed = np.array([0.1, 0.5, 0.9, 0.3])
    expected = np.full(4, 0.2)  # flat efficiency -> nothing to remove
    corrected = mc.flatfield_profile(observed, expected, mode="ratio")
    np.testing.assert_allclose(corrected, observed)


def test_flatfield_ratio_removes_multiplicative_gain():
    # observed exactly tracks the efficiency -> corrected should be flat
    expected = np.array([0.1, 0.2, 0.4, 0.05])
    observed = 3.0 * expected
    corrected = mc.flatfield_profile(observed, expected, mode="ratio")
    np.testing.assert_allclose(corrected, corrected[0] * np.ones_like(corrected))


def test_flatfield_logit_runs_and_stays_in_unit_interval():
    observed = np.array([0.1, 0.5, 0.9])
    expected = np.array([0.2, 0.2, 0.4])
    corrected = mc.flatfield_profile(observed, expected, mode="logit")
    assert np.all((corrected > 0) & (corrected < 1))


def test_flatfield_bad_mode_raises():
    with pytest.raises(ValueError):
        mc.flatfield_profile(np.ones(3), np.ones(3), mode="nope")


# ------------------------------------------------------------------ assess_motif_bias
def test_assess_flat_efficiency_not_confounded():
    observed = np.array([0.1, 0.6, 0.9, 0.4, 0.2])
    expected = np.full(5, 0.25)
    diag = mc.assess_motif_bias(observed, expected, motif="A,0")
    assert diag.cv == pytest.approx(0.0, abs=1e-9)
    assert diag.r2 == pytest.approx(0.0, abs=1e-9)
    assert diag.is_confounded is False


def test_assess_structured_efficiency_flagged_confounded():
    expected = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    observed = expected.copy()  # observed perfectly explained by motif track
    diag = mc.assess_motif_bias(observed, expected, motif="A,0")
    assert diag.cv > 0.05
    assert diag.r2 > 0.9
    assert diag.is_confounded is True


# ------------------------------------------------------------------ integration (real fixture)
_HAVE_FIXTURE = _H5.exists() and _BED.exists() and _FASTA.exists() and (
    _FASTA.with_suffix(_FASTA.suffix + ".fai").exists()
)


@pytest.mark.skipif(not _HAVE_FIXTURE, reason="CTCF demo extract or CHM13v2 FASTA not available")
def test_integration_estimate_profile_and_single_reads():
    # (regions here double as the 'background' for mechanics; science needs distal regions)
    eff = mc.estimate_motif_efficiency(_H5, ["A,0", "CG,0"], _BED, _FASTA, k=3, min_count=1)
    summary = eff.summary()
    usable = summary[summary["global_rate"].notna() & (summary["global_rate"] >= 0)]
    assert not usable.empty
    motif = usable.sort_values("n_contexts").iloc[-1]["motif"]
    assert 0.0 <= eff.global_rate(motif) <= 1.0

    prof = mc.correct_aggregate_profile(
        _H5, motif, _BED, _FASTA, eff, window_size=1000, bin_size=50
    )
    assert {"position", "observed", "expected", "corrected", "valid"}.issubset(prof.columns)
    covered = prof[prof["valid"] > 0]
    assert np.all(np.isfinite(covered["observed"]))
    assert np.all(np.isfinite(covered["corrected"]))

    reads = mc.correct_single_reads(_H5, motif, _BED, _FASTA, eff)
    assert {"read_id", "modified_count", "valid_count", "expected_count", "enrichment"}.issubset(
        reads.columns
    )
    if not reads.empty:
        pos = reads[reads["expected_count"] > 0]
        np.testing.assert_allclose(
            pos["enrichment"].to_numpy(),
            (pos["modified_count"] / pos["expected_count"]).to_numpy(),
            rtol=1e-6,
        )
