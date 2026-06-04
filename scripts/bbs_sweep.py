"""BBS modulus-size sweep: pass rate vs the bit-length of M = p*q.

Blum Blum Shub's quality scales with the modulus size. This sweeps M across
several bit-lengths (the 19-bit point reuses the main evaluation's p=499,
q=547) and runs the full battery on 1M bits at each, using the same per-test
aggregation as test_bits.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from randeval import RandomSequence
from randeval.tests_statistical import full_battery
from randeval.tests_statistical.base import Verdict
from randeval.generators import BlumBlumShub

N = 1_000_000
TARGET_BITS = [19, 31, 41, 63, 81]
OUT = Path(__file__).resolve().parents[2] / "results" / "results_bbs_sweep"
OUT.mkdir(parents=True, exist_ok=True)

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def next_blum_prime(start: int) -> int:
    """Smallest prime >= start that is congruent to 3 mod 4."""
    c = start + ((3 - start) % 4)  # round up to next value ≡ 3 mod 4
    while not is_prime(c):
        c += 4
    return c


def blum_pair(bits: int) -> tuple[int, int]:
    """Find two distinct Blum primes whose product has exactly `bits` bits."""
    if bits == 19:
        return 499, 547  # anchor to the main evaluation config
    p = next_blum_prime(1 << (bits // 2))
    lo = (1 << (bits - 1) + 0) // p + 1
    q = next_blum_prime(max(lo, p + 4))
    while (p * q).bit_length() != bits or q == p:
        q = next_blum_prime(q + 4)
        if q >= (1 << bits) // p:  # overshot; bump p and retry
            p = next_blum_prime(p + 4)
            q = next_blum_prime(max((1 << (bits - 1)) // p + 1, p + 4))
    return p, q


def coprime_seed(m: int) -> int:
    from math import gcd
    s = m // 3 | 1
    while gcd(s, m) != 1 or s <= 1:
        s += 2
    return s


def main() -> None:
    print(f"=== BBS modulus sweep, n={N:,}, full battery ===\n")
    results: dict[str, dict] = {}
    for bits in TARGET_BITS:
        p, q = blum_pair(bits)
        m = p * q
        gen = BlumBlumShub(p=p, q=q, seed=coprime_seed(m))
        actual = m.bit_length()
        seq = RandomSequence.from_generator(gen, N)
        per_test: dict[str, dict] = {}
        for t in full_battery():
            try:
                r = t.run(seq.data)
                per_test[t.name] = {"status": "ok", "passed": r.verdict == Verdict.PASS}
            except Exception as exc:  # noqa: BLE001
                per_test[t.name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        n_ok = sum(1 for v in per_test.values() if v["status"] == "ok")
        n_pass = sum(1 for v in per_test.values() if v["status"] == "ok" and v["passed"])
        pr = (n_pass / n_ok) if n_ok else 0.0
        results[f"{actual}bit"] = {
            "target_bits": bits, "modulus_bits": actual, "p": p, "q": q,
            "n_bits": N, "pass_rate": pr, "n_pass": n_pass, "n_ok": n_ok,
            "n_error": sum(1 for v in per_test.values() if v["status"] == "error"),
            "per_test": per_test,
        }
        (OUT / "bbs_sweep.json").write_text(json.dumps(results, indent=2))
        print(f"  M={actual:2d}bit (p={p}, q={q})  pass {pr:6.1%}  ({n_pass}/{n_ok} ok)")

    print(f"\n  written to {OUT/'bbs_sweep.json'}")


if __name__ == "__main__":
    main()
