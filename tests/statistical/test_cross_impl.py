"""Cross-check against an independent implementation (nistrng).

This guards against shared-spec misreadings: where our value and an unrelated
implementation agree on the same input, a NIST worked-example discrepancy is
the document's erratum, not our bug. (This is how the DFT pi-100 erratum and
the Runs bug were originally told apart.) nistrng is optional; skip if absent.
"""
from __future__ import annotations

import pytest

from randeval.tests_statistical import nist
from . import reference as ref

nistrng = pytest.importorskip("nistrng")
from nistrng import SP800_22R1A_BATTERY  # noqa: E402


def _ng(key, bits):
    res, _ = SP800_22R1A_BATTERY[key].run(bits.astype(int))
    return res.score


def test_frequency_matches_nistrng():
    bits = ref.e_bits(200_000)
    assert nist.FrequencyTest().run(bits).p_value == pytest.approx(_ng("monobit", bits), abs=1e-6)


def test_dft_matches_nistrng():
    # the pi-100 DFT value NIST printed is an erratum; we and nistrng agree.
    bits = ref.pi_100()
    assert nist.SpectralTest().run(bits).p_value == pytest.approx(_ng("dft", bits), abs=1e-6)
