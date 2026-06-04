"""Phase 3: Run all extractors on all generators, save extracted bits to disk.

Loads raw bits from results/bits/{generator}.npy, applies each extractor,
saves to results/extracted/{generator}__{extractor}.npy.
Saves incrementally — safe to re-run after crashes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from randeval import RandomSequence
from randeval.extractors import default_extractors
from randeval.extractors import (
    CryptoMiteToeplitz, CryptoMiteCirculant, CryptoMiteDodis, CryptoMiteTrevisan,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
BITS_DIR = RESULTS_DIR / "bits"
EXTRACTED_DIR = RESULTS_DIR / "extracted"
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)


def get_extractors() -> list:
    exts = default_extractors()
    try:
        exts += [
            CryptoMiteToeplitz(min_entropy=0.8),
            CryptoMiteCirculant(min_entropy=0.8),
            CryptoMiteDodis(min_entropy=0.8),
            CryptoMiteTrevisan(min_entropy=0.8, error=1e-4),
        ]
    except Exception:
        print("  [WARN] CryptoMite not available, skipping those extractors")
    return exts


def safe_filename(name: str) -> str:
    """Sanitise extractor name for use in filenames."""
    return name.replace("/", "_").replace("(", "_").replace(")", "_").replace("=", "").replace(",", "_")


def main() -> None:
    print("=== Phase 3: Extract ===\n")

    meta_path = RESULTS_DIR / "generation_meta.json"
    if not meta_path.exists():
        print("ERROR: Run generate.py first")
        return
    meta = json.loads(meta_path.read_text())

    extractors = get_extractors()
    ext_meta_path = RESULTS_DIR / "extraction_meta.json"

    # load existing metadata for resuming
    ext_meta: dict[str, dict[str, dict]] = {}
    if ext_meta_path.exists():
        ext_meta = json.loads(ext_meta_path.read_text())

    total_gens = len(meta)
    for gi, (gen_name, info) in enumerate(meta.items(), 1):
        bits_path = BITS_DIR / f"{gen_name}.npy"
        if not bits_path.exists():
            print(f"  [{gi}/{total_gens}] {gen_name:25s}  SKIP — no bits")
            continue

        if gen_name not in ext_meta:
            ext_meta[gen_name] = {}

        # an extractor is only "done" if metadata records it AND the .npy
        # actually exists on disk — guards against deleted output files
        # falsely appearing cached
        done = set()
        for ext_name, info in ext_meta[gen_name].items():
            fname = info.get("file")
            if fname and (EXTRACTED_DIR / fname).exists():
                done.add(ext_name)
            elif info.get("output_bits", 0) == 0 and "error" in info:
                # extractor errored last time — keep skipping
                done.add(ext_name)

        # drop stale metadata for missing files so it gets rewritten
        ext_meta[gen_name] = {k: v for k, v in ext_meta[gen_name].items() if k in done}
        needed = [e for e in extractors if e.name not in done]
        if not needed:
            print(f"  [{gi}/{total_gens}] {gen_name:25s}  CACHED ({len(done)} extractors)")
            continue

        bits = np.load(bits_path)
        seq = RandomSequence(bits)
        n_exts = len(needed)

        for ei, ext in enumerate(needed, 1):
            fname = safe_filename(ext.name)
            out_path = EXTRACTED_DIR / f"{gen_name}__{fname}.npy"

            print(f"    [{gi}/{total_gens}] {gen_name:20s} [{ei}/{n_exts}] {ext.name:30s} ", end="", flush=True)
            t0 = time.perf_counter()
            try:
                extracted = seq.extract(ext)
                elapsed = (time.perf_counter() - t0) * 1000

                np.save(out_path, extracted.data)
                ext_meta[gen_name][ext.name] = {
                    "output_bits": len(extracted),
                    "mean": float(extracted.data.mean()),
                    "ext_time_ms": round(elapsed, 1),
                    "file": str(out_path.name),
                }
                print(f"{len(extracted):>8,} bits  ({elapsed:.0f}ms)")

                del extracted

            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                ext_meta[gen_name][ext.name] = {
                    "output_bits": 0,
                    "error": str(e),
                    "ext_time_ms": round(elapsed, 1),
                }
                print(f"ERROR ({elapsed:.0f}ms): {e}")

        # save after each generator
        ext_meta_path.write_text(json.dumps(ext_meta, indent=2))

        ok = sum(1 for v in ext_meta[gen_name].values() if v.get("output_bits", 0) > 0)
        print(f"  [{gi}/{total_gens}] {gen_name:25s}  {ok}/{len(extractors)} extracted")

        del bits, seq

    print(f"\nExtracted bits → {EXTRACTED_DIR}/")
    print(f"Metadata → {ext_meta_path}")


if __name__ == "__main__":
    main()
