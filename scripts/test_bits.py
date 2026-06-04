"""Run the full test battery on a directory of .npy bit files.

Usage:
    python test_bits.py results/bits results/generator_tests.json
    python test_bits.py results/extracted results/extraction_tests.json

    # re-run only specific tests and merge into an existing results file
    # (keeps the other tests' cached verdicts; recomputes the aggregates):
    python test_bits.py results/bits results/generator_tests.json \
        --only "NIST 3: Runs,Dieharder: Parking Lot,Novel: Close Pairs"

Runs tests one at a time with per-test timing + RSS logging, flushed after each
test so the last line before an OOM kill identifies the culprit. Partial
per-test results are written to disk after every test, so a kill never wipes
progress on the current file.

Skips files with fewer than 100 bits.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from randeval import RandomSequence, full_battery
from randeval.tests_statistical.base import Verdict

MIN_TESTABLE = 100

# ru_maxrss is bytes on macOS, KB on Linux
_IS_MAC = platform.system() == "Darwin"
_RSS_DIVISOR = 1024 * 1024 if _IS_MAC else 1024  # -> MB


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RSS_DIVISOR


def save(output_path: Path, results: dict) -> None:
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2))
    tmp.replace(output_path)


def aggregate(per_test: dict, n_bits: int, mean: float, total_ms: float) -> dict:
    """Build the per-file 'ok' summary from its per_test dict."""
    n_ok = sum(1 for v in per_test.values() if v.get("status") == "ok")
    n_pass = sum(1 for v in per_test.values() if v.get("status") == "ok" and v.get("passed"))
    return {
        "status": "ok",
        "n_bits": n_bits,
        "mean": mean,
        "test_time_ms": round(total_ms, 1),
        "pass_rate": (n_pass / n_ok) if n_ok else 0.0,
        "n_pass": n_pass,
        "n_fail": n_ok - n_pass,
        "failed_tests": [t for t, v in per_test.items()
                         if v.get("status") == "ok" and not v.get("passed")],
        "errored_tests": [t for t, v in per_test.items() if v.get("status") == "error"],
        "per_test": per_test,
    }


def run_only(bits_dir: Path, output_path: Path, only: list[str]) -> None:
    """Re-run only tests whose name contains one of `only`, merging into the
    existing results file. Other tests keep their cached verdicts; aggregates
    are recomputed. Use after fixing a handful of tests so the unchanged 50+
    are not needlessly re-run (matters most for the large extracted set).
    """
    if not output_path.exists():
        print(f"ERROR: --only needs an existing {output_path} to merge into")
        sys.exit(1)
    results = json.loads(output_path.read_text())
    battery = full_battery()
    selected = [t for t in battery if any(s in t.name for s in only)]
    if not selected:
        print(f"ERROR: no test names matched {only}")
        sys.exit(1)
    print(f"=== Re-running {len(selected)} test(s) and merging into {output_path} ===")
    for t in selected:
        print(f"    {t.name}")

    npy = sorted(bits_dir.glob("*.npy"))
    for i, path in enumerate(npy, 1):
        name = path.stem
        entry = results.get(name)
        if not entry or entry.get("status") != "ok" or "per_test" not in entry:
            print(f"  [{i}/{len(npy)}] {name:45s}  skip (no prior 'ok' entry)")
            continue
        bits = np.load(path)
        seq = RandomSequence(bits)
        per_test = entry["per_test"]
        t0 = time.perf_counter()
        for t in selected:
            try:
                r = t.run(seq.data)
                per_test[t.name] = {
                    "status": "ok",
                    "p_value": float(r.p_value) if r.p_value is not None else None,
                    "statistic": float(r.statistic) if r.statistic is not None else None,
                    "passed": r.verdict == Verdict.PASS,
                    "time_ms": round((time.perf_counter() - t0) * 1000, 1),
                }
            except Exception as exc:  # noqa: BLE001
                per_test[t.name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        old_pr = entry.get("pass_rate", 0.0)
        results[name] = aggregate(per_test, int(len(bits)),
                                  float(bits.mean()), entry.get("test_time_ms", 0.0))
        save(output_path, results)
        print(f"  [{i}/{len(npy)}] {name:45s}  pass {old_pr:.1%} -> {results[name]['pass_rate']:.1%}",
              flush=True)
        del bits, seq
    print(f"\nmerged {len(selected)} test(s) across {len(npy)} files -> {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the statistical test battery on a directory of .npy bit files"
    )
    ap.add_argument("bits_dir", type=Path)
    ap.add_argument("output_json", type=Path)
    ap.add_argument(
        "--only", default=None,
        help="comma-separated test-name substrings: re-run only matching tests "
             "and merge them into an existing output file (other tests keep their "
             "cached verdicts)",
    )
    args = ap.parse_args()
    bits_dir = args.bits_dir
    output_path = args.output_json

    if not bits_dir.exists():
        print(f"ERROR: {bits_dir} does not exist")
        sys.exit(1)

    if args.only:
        run_only(bits_dir, output_path, [s.strip() for s in args.only.split(",") if s.strip()])
        return

    npy_files = sorted(bits_dir.glob("*.npy"))
    if not npy_files:
        print(f"ERROR: no .npy files in {bits_dir}")
        sys.exit(1)

    battery = full_battery()
    print(f"=== Testing {len(npy_files)} files in {bits_dir}  ({len(battery)} tests each) ===\n")

    results: dict[str, dict] = {}
    if output_path.exists():
        try:
            results = json.loads(output_path.read_text())
        except json.JSONDecodeError:
            print(f"WARN: could not parse existing {output_path}, starting fresh")
            results = {}

    total = len(npy_files)
    tested = 0
    skipped = 0

    for i, npy_path in enumerate(npy_files, 1):
        name = npy_path.stem

        prior = results.get(name)
        if prior:
            status = prior.get("status")
            if status == "ok" and "per_test" in prior and len(prior["per_test"]) == len(battery):
                print(f"  [{i}/{total}] {name:45s}  CACHED  pass={prior['pass_rate']:.1%}")
                continue
            if status == "too_short":
                skipped += 1
                continue
            # status == "ok" without per_test (old format) or partial -> re-run (resume will kick in)

        bits = np.load(npy_path)

        if len(bits) < MIN_TESTABLE:
            results[name] = {"status": "too_short", "n_bits": int(len(bits))}
            save(output_path, results)
            skipped += 1
            print(f"  [{i}/{total}] {name:45s}  too_short ({len(bits)} bits)")
            continue

        print(
            f"  [{i}/{total}] {name:45s}  ({len(bits):>10,} bits)  rss_start={rss_mb():.0f}MB",
            flush=True,
        )

        # mark as partial so a kill mid-file leaves the per-test progress
        results[name] = {
            "status": "partial",
            "n_bits": int(len(bits)),
            "per_test": results.get(name, {}).get("per_test", {}) if prior else {},
        }
        save(output_path, results)

        # inner loop: run tests with per-test checkpointing
        seq = RandomSequence(bits)
        done = results[name]["per_test"]
        t_file0 = time.perf_counter()

        for test in battery:
            tname = test.name
            if tname in done:
                continue

            rss_before = rss_mb()
            t0 = time.perf_counter()
            print(
                f"      ├─ {tname:52s}  rss={rss_before:6.0f}MB  ",
                end="",
                flush=True,
            )

            try:
                r = test.run(seq.data)
                elapsed = (time.perf_counter() - t0) * 1000
                rss_after = rss_mb()
                done[tname] = {
                    "status": "ok",
                    "p_value": float(r.p_value) if r.p_value is not None else None,
                    "statistic": float(r.statistic) if r.statistic is not None else None,
                    "passed": r.verdict == Verdict.PASS,
                    "time_ms": round(elapsed, 1),
                    "rss_delta_mb": round(rss_after - rss_before, 1),
                    "rss_peak_mb": round(rss_after, 1),
                }
                tag = "PASS" if r.verdict == Verdict.PASS else "FAIL"
                print(
                    f"{tag}  t={elapsed:7.0f}ms  Δrss={rss_after - rss_before:+6.1f}MB",
                    flush=True,
                )
            except Exception as exc:
                elapsed = (time.perf_counter() - t0) * 1000
                done[tname] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "time_ms": round(elapsed, 1),
                }
                print(
                    f"ERROR {type(exc).__name__}: {exc}  (t={elapsed:.0f}ms)",
                    flush=True,
                )
                traceback.print_exc(limit=2)

            # checkpoint after every test — survives SIGKILL
            save(output_path, results)

        # aggregate and promote partial -> ok
        total_ms = (time.perf_counter() - t_file0) * 1000
        results[name] = aggregate(done, int(len(bits)), float(bits.mean()), total_ms)
        save(output_path, results)
        tested += 1

        print(
            f"      └─ done  pass={results[name]['pass_rate']:.0%}  "
            f"({total_ms/1000:.1f}s, rss_peak={rss_mb():.0f}MB)\n",
            flush=True,
        )

        del bits, seq

    cached = total - tested - skipped
    print(f"\n{tested} tested, {skipped} skipped, {cached} cached → {output_path}")


if __name__ == "__main__":
    main()
