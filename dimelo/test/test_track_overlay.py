"""Self-contained tests for dimelo.track_overlay (synthetic bigWig, no external data).

Run directly (``python -m dimelo.test.test_track_overlay``) or via pytest. Skips
gracefully if pyBigWig is unavailable.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from dimelo import track_overlay as tov


def _make_bigwig(path: str, chrom: str = "chr1", length: int = 20000) -> None:
    """Write a bigWig whose value at position x equals x (a ramp), so orientation
    and coordinate mapping are trivial to check."""
    import pyBigWig

    bw = pyBigWig.open(path, "w")
    bw.addHeader([(chrom, length)])
    starts = list(range(0, length))
    values = [float(x) for x in starts]  # value == coordinate
    bw.addEntries([chrom] * length, starts, ends=[s + 1 for s in starts], values=values)
    bw.close()


def test_read_windows_center_and_orientation():
    import pyBigWig  # noqa: F401  (import guard -> skip if missing)

    with tempfile.TemporaryDirectory() as d:
        bwp = str(Path(d) / "ramp.bw")
        _make_bigwig(bwp)
        center = 10000
        # A '+' and a '-' region at the same center; ramp value == coordinate.
        regions = [("chr1", center - 5, center + 5, "+"), ("chr1", center - 5, center + 5, "-")]
        tw = tov.read_bigwig_windows(bwp, regions, window_size=100, orientation_aware=True)

        assert tw.matrix.shape == (2, 201)  # inclusive window [-100, 100]
        # positions run -100..100; index of position 0 is 100.
        zero_idx = int(np.where(tw.positions == 0)[0][0])
        # '+' strand: value at center == coordinate == 10000.
        assert abs(tw.matrix[0, zero_idx] - center) < 1e-6
        # '+' strand increases left->right (ramp), so downstream > center.
        assert tw.matrix[0, zero_idx + 10] > tw.matrix[0, zero_idx]
        # '-' strand is flipped: downstream (in region orientation) should now
        # correspond to *lower* genomic coordinate, i.e. decreasing values.
        assert tw.matrix[1, zero_idx + 10] < tw.matrix[1, zero_idx]
        # center value identical regardless of strand.
        assert abs(tw.matrix[1, zero_idx] - center) < 1.0
    print("ok: center + orientation")


def test_binned_matches_perbp_mean():
    import pyBigWig  # noqa: F401

    with tempfile.TemporaryDirectory() as d:
        bwp = str(Path(d) / "ramp.bw")
        _make_bigwig(bwp)
        regions = [("chr1", 9900, 10100, "+")]
        perbp = tov.read_bigwig_windows(bwp, regions, window_size=100)
        binned = tov.read_bigwig_windows(bwp, regions, window_size=100, n_bins=20)
        assert binned.matrix.shape == (1, 20)
        # Mean of the whole per-bp window ~= mean of the binned window.
        assert abs(np.nanmean(perbp.matrix) - np.nanmean(binned.matrix)) < 1.0
    print("ok: binned ~= per-bp mean")


def test_metaprofile_and_pileup():
    # metaprofile grouping
    tw = tov.TrackWindows(
        matrix=np.vstack([np.ones(10), np.ones(10) * 3, np.full(10, np.nan)]),
        positions=np.arange(-5, 5),
        region_ids=np.array(["a", "b", "c"]),
        chromosomes=np.array(["chr1"] * 3),
        starts=np.array([0, 0, 0]),
        ends=np.array([1, 1, 1]),
        strands=np.array(["+"] * 3),
        track_name="t",
        window_size=5,
    )
    mp = tov.metaprofile(tw, groups=["x", "x", "y"], error="sem")
    assert abs(mp[mp.group == "x"]["mean"].iloc[0] - 2.0) < 1e-9  # mean(1,3)=2
    assert np.isnan(mp[mp.group == "y"]["mean"].iloc[0])          # all-NaN group

    # read_pileup: valid-weighted fraction methylated
    data = np.array([[1, 0, 1], [1, 1, 0]], dtype=float)
    val = np.array([[1, 1, 1], [1, 0, 1]], dtype=float)  # read2 pos1 = no-data
    prof = tov.read_pileup(data, val)
    assert abs(prof[0] - 1.0) < 1e-9   # both reads methylated, both valid
    assert abs(prof[1] - 0.0) < 1e-9   # only read1 informative (val=1), and it is unmethylated
    assert abs(prof[2] - 0.5) < 1e-9   # read1=1, read2=0, both valid
    print("ok: metaprofile + pileup")


def test_nucleosome_phasing_recovers_spacing():
    # Synthetic phased dyad profile: cosine with a 190 bp period -> dyad peaks at
    # 0, +/-190, +/-380 ... Metrics should recover ~190 bp spacing and +1 ~ 190.
    period = 190
    positions = np.arange(-500, 501)
    profile = np.cos(2 * np.pi * positions / period)
    tw = tov.TrackWindows(
        matrix=np.tile(profile, (5, 1)),
        positions=positions,
        region_ids=np.array([f"r{i}" for i in range(5)]),
        chromosomes=np.array(["chr1"] * 5),
        starts=np.zeros(5, int),
        ends=np.ones(5, int),
        strands=np.array(["+"] * 5),
        track_name="dyad",
        window_size=500,
    )
    pr = tov.nucleosome_phasing(tw, smooth_bp=None, min_distance_bp=120)["all"]
    assert pr.plus1 is not None and abs(pr.plus1 - period) < 15
    assert pr.minus1 is not None and abs(pr.minus1 + period) < 15
    assert pr.nrl_peaks is not None and abs(pr.nrl_peaks - period) < 15
    assert pr.nrl_autocorr is not None and abs(pr.nrl_autocorr - period) < 20
    print("ok: nucleosome phasing recovers spacing")


if __name__ == "__main__":
    try:
        import pyBigWig  # noqa: F401
    except Exception:
        print("SKIP: pyBigWig not installed")
        raise SystemExit(0)
    test_read_windows_center_and_orientation()
    test_binned_matches_perbp_mean()
    test_metaprofile_and_pileup()
    test_nucleosome_phasing_recovers_spacing()
    print("ALL TRACK_OVERLAY TESTS PASSED")
