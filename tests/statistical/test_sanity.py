"""Internal sanity checks for tests without a published reference value.

A correct battery should pass good random data and flag obviously broken data.
This covers the non-NIST tests (Dieharder, SP800-90B, entropy, distribution,
autocorrelation, novel) where there is no NIST worked example to anchor on.
"""
from __future__ import annotations

import numpy as np
import pytest

from randeval.tests_statistical import full_battery, nist
from randeval.tests_statistical.base import Verdict


def run_battery(data: np.ndarray):
    """Run the whole battery, tolerating per-test exceptions.

    Returns (n_pass, n_fail, errors) where errors maps test name -> message.
    """
    n_pass = n_fail = 0
    errors: dict[str, str] = {}
    for t in full_battery():
        try:
            r = t.run(data)
            if r.verdict == Verdict.PASS:
                n_pass += 1
            else:
                n_fail += 1
        except Exception as ex:  # noqa: BLE001
            errors[t.name] = f"{type(ex).__name__}: {ex}"
    return n_pass, n_fail, errors


def good_stream(n: int = 200_000, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, n, dtype=np.uint8)


def test_good_input_mostly_passes():
    n_pass, n_fail, errors = run_battery(good_stream())
    assert not errors, f"battery raised on good data: {errors}"
    rate = n_pass / (n_pass + n_fail)
    assert rate >= 0.85, f"good data only passed {rate:.0%} of the battery"


def test_all_zeros_mostly_fails():
    data = np.zeros(200_000, dtype=np.uint8)
    n_pass, n_fail, _ = run_battery(data)
    rate = n_fail / (n_pass + n_fail)
    assert rate >= 0.5, f"all-zeros only failed {rate:.0%} of the battery"


def test_all_ones_mostly_fails():
    data = np.ones(200_000, dtype=np.uint8)
    n_pass, n_fail, _ = run_battery(data)
    assert n_fail / (n_pass + n_fail) >= 0.5


# ── specific anchors ──────────────────────────────────────────

def test_frequency_flags_constant():
    assert nist.FrequencyTest().run(np.zeros(10_000, dtype=np.uint8)).verdict == Verdict.FAIL
    assert nist.FrequencyTest().run(np.ones(10_000, dtype=np.uint8)).verdict == Verdict.FAIL


def test_frequency_passes_good():
    assert nist.FrequencyTest().run(good_stream(10_000)).verdict == Verdict.PASS


def test_fast_oscillation_detected():
    # 0101... is balanced (Frequency passes) but oscillates too fast — the Runs
    # test must catch it. Mirrors the glibc-LCG bit-0 weakness in notes.md.
    alt = np.tile([0, 1], 50_000).astype(np.uint8)
    assert nist.FrequencyTest().run(alt).verdict == Verdict.PASS
    assert nist.RunsTest().run(alt).verdict == Verdict.FAIL
