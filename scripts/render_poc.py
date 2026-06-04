"""Render the Cornell box driven by randeval bit streams.

Usage:
    python scripts/render_poc.py --rng MersenneTwister --spp 4 --resolution 16
    python scripts/render_poc.py --all --spp 4 --resolution 16

The 6 RNGs of the chapter are picked by name; "extracted" generators come
from results/results_10mil/extracted/<gen>__<extractor>.npy.
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

# point drjit at the brewed LLVM before any mitsuba import happens
os.environ.setdefault("DRJIT_LIBLLVM_PATH", "/opt/homebrew/opt/llvm/lib/libLLVM.dylib")

from randeval.rendering.mitsuba_adapter import render_cornell_mitsuba  # noqa: E402
from randeval.rendering.integrator import psnr  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BITS_DIR = REPO_ROOT / "results" / "results_10mil" / "bits"
EXT_DIR = REPO_ROOT / "results" / "results_10mil" / "extracted"
OUT_DIR = REPO_ROOT / "results" / "rendering"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# RNGs the chapter compares. The simulated-quantum extractor pair was originally
# Trevisan, but Trevisan is cubic-time and caps the available bits; we keep it
# for parity with the spec but also report Circulant which has 100× more bits.
RNG_SOURCES = {
    "MersenneTwister": BITS_DIR / "MersenneTwister.npy",
    "LCG": BITS_DIR / "LCG.npy",
    "QiskitSimulator": BITS_DIR / "QiskitSimulator.npy",
    "QiskitSimulator_Trevisan": REPO_ROOT / "results" / "rendering" / "QiskitSimulator__Trevisan_300k.npy",
    "QiskitSimulator_Circulant": EXT_DIR / "QiskitSimulator__CryptoMite-Circulant.npy",
    "IBMQuantum": BITS_DIR / "IBMQuantum.npy",
    "IBMQuantum_Circulant": EXT_DIR / "IBMQuantum__CryptoMite-Circulant.npy",
}


def load_bits(name: str) -> np.ndarray:
    p = RNG_SOURCES[name]
    if not p.exists():
        raise FileNotFoundError(f"missing bits for {name}: {p}")
    arr = np.load(p)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def save_png(img: np.ndarray, path: Path) -> None:
    """Linear → sRGB → uint8 PNG via mitsuba.Bitmap."""
    import mitsuba as mi
    bmp = mi.Bitmap(img.astype(np.float32))
    bmp = bmp.convert(
        pixel_format=mi.Bitmap.PixelFormat.RGB,
        component_format=mi.Struct.Type.UInt8,
        srgb_gamma=True,
    )
    bmp.write(str(path))


def save_exr(img: np.ndarray, path: Path) -> None:
    import mitsuba as mi
    mi.Bitmap(img.astype(np.float32)).write(str(path))


def render_one(rng: str, width: int, height: int, spp: int, *, scene: str = "cornell") -> tuple[np.ndarray, dict]:
    bits = load_bits(rng)
    t0 = time.perf_counter()
    img, info = render_cornell_mitsuba(bits, width, height, spp, scene=scene)
    info["rng"] = rng
    info["bits_available"] = int(bits.size)
    info["render_time_s"] = round(time.perf_counter() - t0, 3)
    return img, info


def render_reference(width: int, height: int, spp: int, *, scene: str = "cornell") -> tuple[np.ndarray, dict]:
    t0 = time.perf_counter()
    img, info = render_cornell_mitsuba(None, width, height, spp, scene=scene)
    info["rng"] = "reference_independent"
    info["render_time_s"] = round(time.perf_counter() - t0, 3)
    return img, info


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rng", choices=list(RNG_SOURCES.keys()) + ["all", "reference"], default="all")
    ap.add_argument("--spp", type=int, default=4)
    ap.add_argument("--resolution", type=int, default=16, help="square resolution edge")
    ap.add_argument("--reference-spp", type=int, default=256)
    ap.add_argument("--scene", choices=["cornell", "cornell_glass"], default="cornell")
    ap.add_argument("--write-exr", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    W = H = args.resolution
    scene = args.scene
    # filename namespace tag — empty for default, "_glass" for cornell_glass
    tag = "" if scene == "cornell" else "_glass"

    if args.rng == "reference":
        img, info = render_reference(W, H, args.reference_spp, scene=scene)
        out = OUT_DIR / f"reference{tag}__res{W}_spp{args.reference_spp}.png"
        save_png(img, out)
        if args.write_exr:
            save_exr(img, out.with_suffix(".exr"))
        np.save(out.with_suffix(".npy"), img)
        print(json.dumps(info, indent=2))
        return 0

    # ensure reference exists at this resolution
    ref_npy = OUT_DIR / f"reference{tag}__res{W}_spp{args.reference_spp}.npy"
    if not ref_npy.exists() or _shape_mismatch(ref_npy, W, H):
        print(f"[ref] rendering {scene} {W}x{H} reference at spp={args.reference_spp}…", flush=True)
        ref_img, ref_info = render_reference(W, H, args.reference_spp, scene=scene)
        save_png(ref_img, OUT_DIR / f"reference{tag}__res{W}_spp{args.reference_spp}.png")
        np.save(ref_npy, ref_img)
        print(f"[ref] done in {ref_info['render_time_s']}s", flush=True)
    else:
        ref_img = np.load(ref_npy)

    rngs = list(RNG_SOURCES.keys()) if args.rng == "all" else [args.rng]
    results: dict[str, dict] = {}

    for r in rngs:
        try:
            img, info = render_one(r, W, H, args.spp, scene=scene)
        except Exception as e:
            print(f"[{r}] FAILED: {e!r}", flush=True)
            results[r] = {"error": repr(e)}
            continue
        ps = psnr(ref_img.astype(np.float64), img.astype(np.float64), peak=1.0)
        info["psnr_vs_reference"] = float(ps)
        out = OUT_DIR / f"{r}{tag}__res{W}_spp{args.spp}.png"
        save_png(img, out)
        np.save(out.with_suffix(".npy"), img)
        if args.write_exr:
            save_exr(img, out.with_suffix(".exr"))
        results[r] = info
        print(f"[{r}] PSNR={ps:.2f}  floats={info.get('floats_consumed','?')}  "
              f"wraps={info.get('wraps','?')}  time={info['render_time_s']}s", flush=True)

    # update psnr table — namespaced by resolution and scene
    table_path = OUT_DIR / f"psnr_table{tag}_res{W}.json"
    table = json.loads(table_path.read_text()) if table_path.exists() else {}
    for r, info in results.items():
        if "error" in info:
            continue
        cell = table.setdefault(r, {})
        cell[f"spp{args.spp}"] = info["psnr_vs_reference"]
        cell.setdefault("meta", {})[f"spp{args.spp}"] = {
            "floats_consumed": info.get("floats_consumed"),
            "wraps": info.get("wraps"),
            "bits_available": info.get("bits_available"),
            "render_time_s": info.get("render_time_s"),
            "resolution": W,
        }
    table_path.write_text(json.dumps(table, indent=2))
    print(f"\nupdated {table_path}")
    return 0


def _shape_mismatch(npy_path: Path, w: int, h: int) -> bool:
    try:
        arr = np.load(npy_path, mmap_mode="r")
        return arr.shape[:2] != (h, w)
    except Exception:
        return True


if __name__ == "__main__":
    sys.exit(main())
