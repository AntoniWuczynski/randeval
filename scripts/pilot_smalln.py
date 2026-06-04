"""Small-n pilot: run the full battery at n=100 bits per generator.

Mirrors the main pipeline (generate.py configs, test_bits.py aggregation) but
at the MIN_TESTABLE floor of 100 bits, to show how the battery behaves when
there is almost no data to work with. Reproducible offline subset only:
the 11 classical, 6 CSPRNG and 5 simulated-quantum generators (no hardware,
network or IBM sources).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from randeval import RandomSequence
from randeval.tests_statistical import full_battery
from randeval.tests_statistical.base import Verdict
from randeval.generators import (
    LCG, MersenneTwister, PCG64, Philox, SFC64,
    Xorshift128Plus, Xoshiro256StarStar, LFSR,
    MiddleSquare, MiddleSquareWeyl, WichmannHill,
    SystemRandom, HMAC_DRBG, Hash_DRBG, BlumBlumShub,
    ChaCha20, AESCTR_DRBG,
    QiskitSimulator, MultiQubitHadamard,
    EntanglementBasedQRNG, QuantumPhaseEstimationQRNG,
    RandomRotationQRNG,
)

N = 100
OUT = Path(__file__).resolve().parents[2] / "results" / "results_pilot_100"
OUT.mkdir(parents=True, exist_ok=True)


def gens():
    return [
        ("LCG", "Classical", LCG()),
        ("MersenneTwister", "Classical", MersenneTwister(seed=42)),
        ("PCG64", "Classical", PCG64(seed=42)),
        ("Philox", "Classical", Philox(seed=42)),
        ("SFC64", "Classical", SFC64(seed=42)),
        ("Xorshift128+", "Classical", Xorshift128Plus()),
        ("Xoshiro256**", "Classical", Xoshiro256StarStar()),
        ("LFSR", "Classical", LFSR()),
        ("MiddleSquare", "Classical", MiddleSquare()),
        ("MiddleSquareWeyl", "Classical", MiddleSquareWeyl()),
        ("WichmannHill", "Classical", WichmannHill()),
        ("SystemRandom", "CSPRNG", SystemRandom()),
        ("HMAC_DRBG", "CSPRNG", HMAC_DRBG(seed=b"testseed12345678901234567890abcd")),
        ("Hash_DRBG", "CSPRNG", Hash_DRBG(seed=b"testseed12345678901234567890abcd")),
        ("BlumBlumShub", "CSPRNG", BlumBlumShub(p=499, q=547, seed=12345)),
        ("ChaCha20", "CSPRNG", ChaCha20()),
        ("AES-CTR-DRBG", "CSPRNG", AESCTR_DRBG()),
        ("QiskitSimulator", "Quantum-Sim", QiskitSimulator()),
        ("MultiQubitHadamard", "Quantum-Sim", MultiQubitHadamard(num_qubits=8)),
        ("EntanglementQRNG", "Quantum-Sim", EntanglementBasedQRNG()),
        ("QPE-QRNG", "Quantum-Sim", QuantumPhaseEstimationQRNG()),
        ("RandomRotation(pi4)", "Quantum-Sim", RandomRotationQRNG(theta=0.7854)),
    ]


def main() -> None:
    print(f"=== Small-n pilot: N={N} bits, full battery ===\n")
    results: dict[str, dict] = {}
    for i, (name, cat, gen) in enumerate(gens(), 1):
        gen.reset()
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
        n_err = sum(1 for v in per_test.values() if v["status"] == "error")
        pr = (n_pass / n_ok) if n_ok else 0.0
        results[name] = {
            "category": cat, "n_bits": N, "pass_rate": pr,
            "n_pass": n_pass, "n_ok": n_ok, "n_error": n_err,
            "per_test": per_test,
        }
        (OUT / "pilot.json").write_text(json.dumps(results, indent=2))
        print(f"  [{i:2d}/22] {name:22s} {cat:12s} pass {pr:6.1%}  ({n_pass}/{n_ok} ok, {n_err} err)")

    prs = [r["pass_rate"] for r in results.values()]
    print(f"\n  written to {OUT/'pilot.json'}")
    print(f"  pass-rate range {min(prs):.1%}-{max(prs):.1%}, mean {sum(prs)/len(prs):.1%}")


if __name__ == "__main__":
    main()
