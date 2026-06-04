"""Null-calibration: a correct test must not flag good random data.

Under the null hypothesis (a high-quality source) a proper hypothesis test
should produce P-values ~ Uniform(0,1) and therefore reject at roughly
alpha = 0.01. A test that systematically rejects good data is miscalibrated —
this is exactly how the three documented bugs (Parking Lot, Permutation
Entropy, Adaptive Proportion) manifested at large n.

Streams come from NumPy's PCG64 with fixed seeds so the suite is reproducible
(no flaky CI) while still being high quality. The heavier 200-stream / multi-
size sweep over all 68 tests lives in ``run_calibration.py``.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sp_stats

from randeval.tests_statistical import nist
from randeval.tests_statistical.base import Verdict
from randeval.tests_statistical.dieharder import BirthdaySpacingsTest, ParkingLotTest
from randeval.tests_statistical.entropy import PermutationEntropyTest
from randeval.tests_statistical.sp800_90b import AdaptiveProportionTest


def streams(n_bits: int, count: int, seed0: int = 1000):
    """Yield ``count`` independent PCG64 bit streams of length ``n_bits``."""
    for k in range(count):
        rng = np.random.default_rng(seed0 + k)
        yield rng.integers(0, 2, n_bits, dtype=np.uint8)


def reject_and_ks(test_factory, n_bits: int, count: int):
    """Run a test over many null streams; return (reject_rate, ks_uniform_p)."""
    ps = np.array([test_factory().run(s).p_value for s in streams(n_bits, count)])
    rejects = float(np.mean(ps < 0.01))
    ks_p = float(sp_stats.kstest(ps, "uniform").pvalue)
    return rejects, ks_p


# Proper NIST hypothesis tests that should give Uniform(0,1) p-values at 1e5.
PROPER = {
    "frequency": lambda: nist.FrequencyTest(),
    "block_frequency": lambda: nist.BlockFrequencyTest(),
    "runs": lambda: nist.RunsTest(),
    "dft": lambda: nist.SpectralTest(),
    "cusum_fwd": lambda: nist.CumulativeSumsTest(forward=True),
    "approx_entropy": lambda: nist.ApproximateEntropyTest(block_length=8),
}


@pytest.mark.parametrize("name", list(PROPER))
def test_proper_tests_calibrated(name: str):
    reject, ks_p = reject_and_ks(PROPER[name], 100_000, 40)
    # a correct test rejects ~1% of good streams; allow head-room for 40 draws
    assert reject <= 0.10, f"{name} over-rejects good data: {reject:.0%}"
    # p-values should not be grossly non-uniform
    assert ks_p > 1e-3, f"{name} p-values non-uniform (KS p={ks_p:.4f})"


# ── The three previously-miscalibrated tests ──────────────────

def test_permutation_entropy_calibrated():
    reject, ks_p = reject_and_ks(lambda: PermutationEntropyTest(), 1_000_000, 40)
    assert reject <= 0.10, f"Permutation Entropy over-rejects: {reject:.0%}"
    assert ks_p > 1e-3


def test_adaptive_proportion_calibrated():
    reject, ks_p = reject_and_ks(lambda: AdaptiveProportionTest(), 1_000_000, 40)
    assert reject <= 0.10, f"Adaptive Proportion over-rejects: {reject:.0%}"


def test_parking_lot_calibrated():
    # Parking Lot needs >= 768,000 bits per rep; test at 1e6.
    reject, ks_p = reject_and_ks(lambda: ParkingLotTest(), 1_000_000, 20)
    assert reject <= 0.10, f"Parking Lot over-rejects: {reject:.0%}"


def test_birthday_spacings_calibrated():
    # was over-rejecting ~30% with year_length=2^18 (n ~ m^2, Poisson regime
    # invalid). Fixed to n=2^24 (n >> m^2). Discrete statistic, so check
    # rejection rate only, not KS-uniformity.
    reject, _ = reject_and_ks(lambda: BirthdaySpacingsTest(), 200_000, 40)
    assert reject <= 0.10, f"Birthday Spacings over-rejects: {reject:.0%}"


def test_close_pairs_calibrated():
    # was over-rejecting ~10% at all sizes (KS on dependent NN distances +
    # boundary bias). Reworked to a toroidal Poisson close-pair count.
    from randeval.tests_statistical.novel import ClosePairsTest
    reject, _ = reject_and_ks(lambda: ClosePairsTest(), 100_000, 40)
    assert reject <= 0.10, f"Close Pairs over-rejects: {reject:.0%}"


def test_parking_lot_insufficient_data_does_not_fail():
    # below the bit budget it must not spuriously reject
    res = ParkingLotTest().run(np.random.default_rng(0).integers(0, 2, 100_000, dtype=np.uint8))
    assert res.verdict == Verdict.PASS
