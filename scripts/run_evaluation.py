"""Run the full evaluation pipeline: generate → test → extract+test.

Each phase is idempotent — it skips work already saved to disk.
If any phase crashes, re-run this script and it picks up where it left off.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
# use the FYP-level venv which has all deps (qiskit, cryptography, cryptomite, etc.)
PYTHON = str(SCRIPTS_DIR.parent.parent / ".venv" / "bin" / "python")


def run_phase(name: str, script: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}\n")
    t0 = time.perf_counter()
    parts = script.split()
    cmd = [PYTHON, str(SCRIPTS_DIR / parts[0])] + parts[1:]
    result = subprocess.run(cmd, cwd=str(SCRIPTS_DIR.parent))
    elapsed = time.perf_counter() - t0
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    if result.returncode == 0:
        print(f"\n  ✓ {name} completed in {mins}m {secs}s")
        return True
    else:
        print(f"\n  ✗ {name} FAILED (exit code {result.returncode}) after {mins}m {secs}s")
        print(f"    Saved results are safe. Fix the issue and re-run.")
        return False


def main() -> None:
    print("randeval — Full Evaluation Pipeline")
    print("Results are saved incrementally. Safe to re-run after crashes.\n")

    phases = [
        ("Phase 1: Generate bit sequences", "generate.py"),
        ("Phase 2: Test raw generators", "test_bits.py results/bits results/generator_tests.json"),
        ("Phase 3: Extract", "extract.py"),
        ("Phase 4: Test extractions", "test_bits.py results/extracted results/extraction_tests.json"),
    ]

    t_start = time.perf_counter()
    for name, script in phases:
        if not run_phase(name, script):
            print("\nPipeline stopped. Fix the error above and re-run.")
            sys.exit(1)

    total = time.perf_counter() - t_start
    mins = int(total // 60)
    secs = int(total % 60)
    print(f"\n{'=' * 60}")
    print(f"  All phases complete in {mins}m {secs}s")
    print(f"  Results in: results/")
    print(f"    generation_meta.json")
    print(f"    generator_tests.json")
    print(f"    extraction_tests.json")
    print(f"    bits/*.npy")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
