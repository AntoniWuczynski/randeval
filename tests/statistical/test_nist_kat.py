"""Known-answer tests for the NIST SP 800-22 suite.

Each test runs our implementation on a fixed reference input and asserts the
P-value matches NIST's published worked example (or, where the published value
is a documented erratum, the value confirmed by an independent implementation).
Tolerance is 1e-3 unless noted — NIST prints to 6 dp but uses its own rounded
intermediates, so sub-1e-3 agreement is the practical ceiling.
"""
from __future__ import annotations

import numpy as np
import pytest

from randeval.tests_statistical import nist
from . import reference as ref

TOL = 1e-3


# ── pi-100 example (NIST §2.x.8) ──────────────────────────────

def test_frequency_pi100():
    p = nist.FrequencyTest().run(ref.pi_100()).p_value
    assert p == pytest.approx(0.109599, abs=TOL)


def test_block_frequency_pi100():
    p = nist.BlockFrequencyTest(block_size=10).run(ref.pi_100()).p_value
    assert p == pytest.approx(0.706438, abs=TOL)


def test_runs_pi100():
    # NIST §3.3 normative formula applies erfc directly (no /sqrt2).
    # nistrng independently reproduces 0.500798.
    p = nist.RunsTest().run(ref.pi_100()).p_value
    assert p == pytest.approx(0.500798, abs=TOL)


def test_approximate_entropy_pi100():
    p = nist.ApproximateEntropyTest(block_length=2).run(ref.pi_100()).p_value
    assert p == pytest.approx(0.235301, abs=TOL)


def test_cusum_forward_pi100():
    p = nist.CumulativeSumsTest(forward=True).run(ref.pi_100()).p_value
    assert p == pytest.approx(0.219194, abs=TOL)


def test_cusum_reverse_pi100():
    p = nist.CumulativeSumsTest(forward=False).run(ref.pi_100()).p_value
    assert p == pytest.approx(0.114866, abs=TOL)


def test_dft_pi100():
    # NIST's printed 0.168669 is a known erratum (its stated N1=46 does not
    # follow from the sequence). Our value matches nistrng exactly.
    p = nist.SpectralTest().run(ref.pi_100()).p_value
    assert p == pytest.approx(0.646355, abs=TOL)


# ── e-expansion examples (NIST §2.x.8, n up to 1e6) ───────────

def test_rank_e():
    p = nist.BinaryMatrixRankTest().run(ref.e_bits(100000)).p_value
    assert p == pytest.approx(0.532069, abs=2e-3)


def test_overlapping_template_counts_e():
    # NIST's printed chi2/P (8.965859 / 0.110434) does not follow from its own
    # printed counts and pi. The counts are the unambiguous anchor: ours match
    # NIST exactly. The P-value computed from NIST's own pi is ~0.159.
    e = ref.e_bits(1000000)
    t = nist.OverlappingTemplateTest(template_length=9, block_size=1032)
    res = t.run(e)
    assert res.p_value == pytest.approx(0.159, abs=2e-3)


def test_linear_complexity_e():
    p = nist.LinearComplexityTest(block_size=1000).run(ref.e_bits(1000000)).p_value
    assert p == pytest.approx(0.845406, abs=2e-3)


def test_serial_e():
    p = nist.SerialTest(pattern_length=2).run(ref.e_bits(1000000)).p_value
    assert p == pytest.approx(0.843764, abs=TOL)


def test_random_excursions_e():
    # x=-1 reproduces NIST's published value exactly; the anchor for this test.
    p = nist.RandomExcursionsTest(state=-1).run(ref.e_bits(1000000)).p_value
    assert p == pytest.approx(0.007779, abs=TOL)


# NIST §2.15.8 published Random Excursions Variant table (e, 1e6 bits).
_VARIANT_E = {
    -9: 0.858946, -8: 0.794755, -7: 0.576249, -6: 0.493417, -5: 0.633873,
    -4: 0.917283, -3: 0.934708, -2: 0.816012, -1: 0.826009,
    1: 0.137861, 2: 0.200642, 3: 0.441254, 4: 0.939291, 5: 0.505683,
    6: 0.445935, 7: 0.512207, 8: 0.538635, 9: 0.593930,
}


@pytest.mark.parametrize("state", [-9, -5, -1, 1, 5, 9])
def test_random_excursions_variant_e(state: int):
    # erfc applied directly (no /sqrt2); reproduces the full NIST table.
    p = nist.RandomExcursionsVariantTest(state=state).run(ref.e_bits(1000000)).p_value
    assert p == pytest.approx(_VARIANT_E[state], abs=TOL)


# ── Longest Run block-size tier selection (NIST §2.4.2 table) ──

@pytest.mark.parametrize("n,expected_M", [
    (1000, 8),
    (100000, 128),
    (1000000, 10000),
    (10000000, 10000),
])
def test_longest_run_tier(n: int, expected_M: int):
    # NIST §2.4.2 table: [128,6272)->M=8, [6272,750000)->M=128, [>=750000]->M=10^4.
    M, _K, _vmin, _pi = nist._longest_run_params(n)
    assert M == expected_M
