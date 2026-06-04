"""Phase 1: Generate bit sequences from all generators and save to disk.

Generators are grouped into categories. Each has different requirements:

  Classical (11):     No dependencies. Pure algorithmic PRNGs.
  CSPRNG (6):         ChaCha20/AES need `cryptography` package.
  Quantum-Sim (5):    Need `qiskit` package (pip install randeval[quantum]).
  Quantum-Real (1):   IBM hardware. Needs IBM_TOKEN env var or FYP/.env.
                      Uses queue — may take minutes. Limited credits.
  TRNG-Network (3):   RandomOrg, ANUQRNG, HotBits. Need `requests` + internet.
                      Subject to API rate limits. RandomOrg/HotBits have optional
                      API keys for higher quotas. At 1M bits, expect slowness
                      or partial data from free tiers.
  TRNG-Hardware (2):  RDRAND/RDSEED need x86_64 Intel/AMD CPU with those
                      instructions + `rdrand` Python package.
                      Will NOT work on Apple Silicon.
  TRNG-Device (1):    HWRNGDevice reads /dev/hwrng. Linux only.
  TRNG-Local (3):     TimingJitter (any OS, CPU busy-loop — slow),
                      AudioNoise (needs `sounddevice` + microphone),
                      CameraNoiseLavaRand (needs `opencv-python` + camera).

Generators that fail to initialise or generate are skipped with a message.
Results are saved incrementally — safe to re-run after crashes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ensure randeval is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from randeval import RandomSequence
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from randeval.generators import (
    # classical
    LCG, MersenneTwister, PCG64, Philox, SFC64,
    Xorshift128Plus, Xoshiro256StarStar, LFSR,
    MiddleSquare, MiddleSquareWeyl, WichmannHill,
    # csprng
    SystemRandom, HMAC_DRBG, Hash_DRBG, BlumBlumShub,
    ChaCha20, AESCTR_DRBG,
    # quantum
    QiskitSimulator, MultiQubitHadamard,
    EntanglementBasedQRNG, QuantumPhaseEstimationQRNG,
    RandomRotationQRNG, IBMQuantumBackend,
    # trng
    RandomOrgGenerator, ANUQRNGGenerator, HotBitsGenerator,
    RDRANDGenerator, RDSEEDGenerator, HWRNGDevice,
    TimingJitterGenerator, AudioNoiseGenerator, CameraNoiseLavaRand,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
BITS_DIR = RESULTS_DIR / "bits"
BITS_DIR.mkdir(parents=True, exist_ok=True)

# The main runs use two tiers, 1M and 10M bits. Everything scales off BIG_N:
# the IBM cap is BIG_N/20 (50k at 1M, 500k at 10M) and the TRNG caps are
# BIG_N/10 (API rate limits / slow local sources make the full count impractical).
BIG_N = 1000000
QUANTUM_REAL_N = BIG_N // 20
TRNG_NETWORK_N = BIG_N // 10
TRNG_LOCAL_N = BIG_N // 10


def _try_init(name: str, factory, *args, **kwargs):
    """Try to instantiate a generator, return None on failure."""
    try:
        return factory(*args, **kwargs)
    except Exception as e:
        print(f"  [SKIP] {name} — {e}")
        return None


def get_generators() -> list[tuple[str, str, object, int]]:
    """Build the full generator list, skipping anything that can't init.

    Each entry is (display_name, category, generator_instance, n_bits).
    Generators that need missing hardware/packages/keys are silently skipped.
    """
    gens: list[tuple[str, str, object, int]] = [
        # --- Classical PRNGs (always available) ---
        ("LCG", "Classical", LCG(), BIG_N),
        ("MersenneTwister", "Classical", MersenneTwister(seed=42), BIG_N),
        ("PCG64", "Classical", PCG64(seed=42), BIG_N),
        ("Philox", "Classical", Philox(seed=42), BIG_N),
        ("SFC64", "Classical", SFC64(seed=42), BIG_N),
        ("Xorshift128+", "Classical", Xorshift128Plus(), BIG_N),
        ("Xoshiro256**", "Classical", Xoshiro256StarStar(), BIG_N),
        ("LFSR", "Classical", LFSR(), BIG_N),
        ("MiddleSquare", "Classical", MiddleSquare(), BIG_N),
        ("MiddleSquareWeyl", "Classical", MiddleSquareWeyl(), BIG_N),
        ("WichmannHill", "Classical", WichmannHill(), BIG_N),
        # --- CSPRNGs ---
        ("SystemRandom", "CSPRNG", SystemRandom(), BIG_N),
        ("HMAC_DRBG", "CSPRNG", HMAC_DRBG(seed=b"testseed12345678901234567890abcd"), BIG_N),
        ("Hash_DRBG", "CSPRNG", Hash_DRBG(seed=b"testseed12345678901234567890abcd"), BIG_N),
        ("BlumBlumShub", "CSPRNG", BlumBlumShub(p=499, q=547, seed=12345), BIG_N),
        ("ChaCha20", "CSPRNG", ChaCha20(), BIG_N),
        ("AES-CTR-DRBG", "CSPRNG", AESCTR_DRBG(), BIG_N),
        # --- Quantum (simulated — needs qiskit) ---
        ("QiskitSimulator", "Quantum-Sim", QiskitSimulator(), BIG_N),
        ("MultiQubitHadamard", "Quantum-Sim", MultiQubitHadamard(num_qubits=8), BIG_N),
        ("EntanglementQRNG", "Quantum-Sim", EntanglementBasedQRNG(), BIG_N),
        ("QPE-QRNG", "Quantum-Sim", QuantumPhaseEstimationQRNG(), BIG_N),
        ("RandomRotation(pi4)", "Quantum-Sim", RandomRotationQRNG(theta=0.7854), BIG_N),
    ]

    # --- Quantum (real hardware — needs IBM_TOKEN) ---
    token = os.environ.get("IBM_TOKEN", "")
    if token:
        gen = _try_init("IBMQuantum", IBMQuantumBackend, token=token, backend_name="")
        if gen:
            gens.append(("IBMQuantum", "Quantum-Real", gen, QUANTUM_REAL_N))
    else:
        print("  [SKIP] IBMQuantum — set IBM_TOKEN env var or add to FYP/.env")

    # --- TRNG: Network APIs (need requests + internet) ---
    gen = _try_init("RandomOrg", RandomOrgGenerator)
    if gen:
        gens.append(("RandomOrg", "TRNG-Network", gen, TRNG_NETWORK_N))

    gen = _try_init("ANUQRNG", ANUQRNGGenerator)
    if gen:
        gens.append(("ANUQRNG", "TRNG-Network", gen, TRNG_NETWORK_N))

    gen = _try_init("HotBits", HotBitsGenerator)
    if gen:
        gens.append(("HotBits", "TRNG-Network", gen, TRNG_NETWORK_N))

    # --- TRNG: CPU instructions (need x86_64 + rdrand package) ---
    gen = _try_init("RDRAND", RDRANDGenerator)
    if gen:
        gens.append(("RDRAND", "TRNG-Hardware", gen, BIG_N))

    gen = _try_init("RDSEED", RDSEEDGenerator)
    if gen:
        gens.append(("RDSEED", "TRNG-Hardware", gen, BIG_N))

    # --- TRNG: Device (Linux /dev/hwrng) ---
    gen = _try_init("HWRNGDevice", HWRNGDevice)
    if gen:
        gens.append(("HWRNGDevice", "TRNG-Device", gen, BIG_N))

    # --- TRNG: Local hardware (timing, audio, camera) ---
    gen = _try_init("TimingJitter", TimingJitterGenerator)
    if gen:
        gens.append(("TimingJitter", "TRNG-Local", gen, TRNG_LOCAL_N))

    gen = _try_init("AudioNoise", AudioNoiseGenerator)
    if gen:
        gens.append(("AudioNoise", "TRNG-Local", gen, TRNG_LOCAL_N))

    gen = _try_init("CameraLavaRand", CameraNoiseLavaRand)
    if gen:
        gens.append(("CameraLavaRand", "TRNG-Local", gen, TRNG_LOCAL_N))

    return gens


def main() -> None:
    global BIG_N, QUANTUM_REAL_N, TRNG_NETWORK_N, TRNG_LOCAL_N, RESULTS_DIR, BITS_DIR
    ap = argparse.ArgumentParser(description="Phase 1: generate bit sequences.")
    ap.add_argument("--bits", type=int, default=BIG_N,
                    help="bits per classical/CSPRNG/quantum-sim source (default 1e6; use 10000000 for the 10M tier)")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR,
                    help="results directory (default randeval/results)")
    args = ap.parse_args()
    BIG_N = args.bits
    QUANTUM_REAL_N = BIG_N // 20
    TRNG_NETWORK_N = BIG_N // 10
    TRNG_LOCAL_N = BIG_N // 10
    RESULTS_DIR = args.out
    BITS_DIR = RESULTS_DIR / "bits"
    BITS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Phase 1: Generate ===")
    print(f"  BIG_N={BIG_N:,}  QUANTUM_REAL_N={QUANTUM_REAL_N:,}")
    print(f"  TRNG_NETWORK_N={TRNG_NETWORK_N:,}  TRNG_LOCAL_N={TRNG_LOCAL_N:,}\n")

    existing = {p.stem for p in BITS_DIR.glob("*.npy")}
    meta: dict[str, dict] = {}
    meta_path = RESULTS_DIR / "generation_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    gens = get_generators()
    print(f"\n  {len(gens)} generators to process\n")

    for name, cat, gen, n in gens:
        if name in existing:
            print(f"  {name:25s}  CACHED")
            continue

        gen.reset()
        t0 = time.perf_counter()
        try:
            seq = RandomSequence.from_generator(gen, n)
        except Exception as e:
            print(f"  {name:25s}  ERROR: {e}")
            continue
        elapsed = (time.perf_counter() - t0) * 1000

        np.save(BITS_DIR / f"{name}.npy", seq.data)
        meta[name] = {
            "category": cat,
            "n_bits": len(seq),
            "mean": float(seq.data.mean()),
            "gen_time_ms": round(elapsed, 1),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"  {name:25s}  {cat:14s}  n={len(seq):>9,}  mean={seq.data.mean():.4f}  gen={elapsed:.0f}ms")

    print(f"\n{len(meta)} generators saved to {BITS_DIR}/")


if __name__ == "__main__":
    main()
