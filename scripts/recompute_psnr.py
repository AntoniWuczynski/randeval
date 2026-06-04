"""Recompute clipped + peak-HDR PSNR for every rendered cell from its .npy.

The headline metric is "clipped" PSNR: clip both linear images to [0, 1] and
use peak = 1.0. peak-HDR uses peak = max(reference). This recomputes both for
every cell that has a saved render and writes them back into the psnr_table
JSONs, so the top-level cell value is uniformly the clipped metric (some cells
previously held the raw peak=1.0 value, which mixed metrics across the SPP
sweep).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results" / "rendering"


def psnr(ref: np.ndarray, test: np.ndarray, peak: float, clip: bool) -> float:
    a, b = (np.clip(ref, 0, 1), np.clip(test, 0, 1)) if clip else (ref, test)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-20 else 10.0 * math.log10(peak * peak / mse)


def reference_for(res: int, scene_tag: str) -> np.ndarray:
    cands = sorted(OUT.glob(f"reference{scene_tag}__res{res}_spp*.npy"),
                   key=lambda p: int(p.stem.split("_spp")[-1]))
    return np.load(cands[-1]).astype(np.float64)


def patch(table_file: str, scene_tag: str, fixed_res: int | None) -> None:
    path = OUT / table_file
    if not path.exists():
        return
    blob = json.loads(path.read_text())
    refs: dict[int, np.ndarray] = {}
    for rng, cell in blob.items():
        # nested-by-res (master) vs flat single-res (per-res file)
        res_cells = ([(int(k[3:]), v) for k, v in cell.items() if k.startswith("res")]
                     if fixed_res is None else [(fixed_res, cell)])
        for res, c in res_cells:
            for spp_key in [k for k in c if k.startswith("spp")]:
                spp = int(spp_key[3:])
                npy = OUT / f"{rng}{scene_tag}__res{res}_spp{spp}.npy"
                if not npy.exists():
                    continue
                if res not in refs:
                    refs[res] = reference_for(res, scene_tag)
                ref = refs[res]
                meta = c.setdefault("meta", {}).setdefault(spp_key, {})
                wraps = meta.get("wraps")
                if wraps is not None and wraps > 4:
                    # stream cycled too many times to be a fair RNG sample
                    c[spp_key] = None
                    meta["psnr_clipped"] = None
                    meta["psnr_peakHDR"] = None
                    continue
                test = np.load(npy).astype(np.float64)
                clipped = round(psnr(ref, test, 1.0, clip=True), 4)
                peakhdr = round(psnr(ref, test, float(ref.max()), clip=False), 4)
                c[spp_key] = clipped
                meta["psnr_clipped"] = clipped
                meta["psnr_peakHDR"] = peakhdr
    path.write_text(json.dumps(blob, indent=2))
    print(f"  patched {table_file}")


def main() -> None:
    print("Recomputing clipped/peakHDR PSNR from renders:")
    patch("psnr_table.json", "", None)
    patch("psnr_table_res16.json", "", 16)
    patch("psnr_table_res32.json", "", 32)
    patch("psnr_table_res64.json", "", 64)
    patch("psnr_table_glass.json", "_glass", None)
    patch("psnr_table_glass_res16.json", "_glass", 16)

    # report the diffuse 16x16 SPP sweep for the chapter
    m = json.loads((OUT / "psnr_table.json").read_text())
    print("\nDiffuse 16x16 SPP sweep (clipped PSNR):")
    for rng, cell in m.items():
        r16 = cell.get("res16", {})
        row = {k: r16.get(k) for k in ("spp1", "spp4", "spp16", "spp64")}
        print(f"  {rng:28s} {row}")


if __name__ == "__main__":
    main()
