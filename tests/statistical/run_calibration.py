"""Heavy tiered null-calibration sweep over the full 68-test battery.

Not part of the fast pytest run — invoke directly for the thorough sweep:

    uv run python -m tests.statistical.run_calibration

Tiers (chosen so the whole thing finishes in well under an hour rather than
the days a flat 200 x 1e7 sweep would take):

  * 1e5 bits, 200 streams  — full battery; the cheap, high-power tier
  * 1e6 bits,  40 streams  — full battery
  * 1e7 bits,  10 streams  — cheap tests + the three former suspects only

For every test it reports the rejection rate on good os.urandom data (should be
~alpha=0.01) and a KS-vs-Uniform p-value. Any test rejecting well above alpha
is miscalibrated.
"""
from __future__ import annotations

import os

import numpy as np
from scipy import stats as sp_stats

from randeval.tests_statistical import full_battery
from randeval.tests_statistical.base import Verdict

# tests cheap enough to run at 1e7
_CHEAP_AT_10M = (
    "Frequency", "Runs", "DFT", "Cumulative", "Parking Lot",
    "Permutation Entropy", "Adaptive Proportion", "Chi-Squared", "Autocorrelation",
)


def urandom_stream(n_bits: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(os.urandom(n_bits // 8), dtype=np.uint8)).astype(np.uint8)


def sweep(n_bits: int, n_streams: int, only: tuple[str, ...] | None = None) -> None:
    pvals: dict[str, list[float]] = {}
    fails: dict[str, int] = {}
    errs: dict[str, str] = {}
    for _ in range(n_streams):
        s = urandom_stream(n_bits)
        for t in full_battery():
            if only is not None and not any(k in t.name for k in only):
                continue
            try:
                r = t.run(s)
                pvals.setdefault(t.name, []).append(float(r.p_value))
                fails[t.name] = fails.get(t.name, 0) + (1 if r.verdict == Verdict.FAIL else 0)
            except Exception as ex:  # noqa: BLE001
                errs[t.name] = f"{type(ex).__name__}: {ex}"

    print(f"\n=== n_bits={n_bits:,}  streams={n_streams} ===")
    print(f"{'test':45s} {'reject':>8s} {'KS-unif':>8s}")
    for name in sorted(pvals):
        ps = np.array(pvals[name])
        ks = sp_stats.kstest(ps, "uniform").pvalue if len(ps) > 4 else float("nan")
        flag = "  <-- HIGH" if fails[name] / len(ps) > 0.05 else ""
        print(f"{name:45s} {fails[name] / len(ps):>7.1%} {ks:>8.3f}{flag}")
    for name, e in errs.items():
        print(f"  ERROR {name}: {e}")


def main() -> None:
    sweep(100_000, 200)
    sweep(1_000_000, 40)
    sweep(10_000_000, 10, only=_CHEAP_AT_10M)


if __name__ == "__main__":
    main()
