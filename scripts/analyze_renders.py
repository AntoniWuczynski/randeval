"""Walk results/rendering/, compute FLIP / error images / spectrum for every cell.

Reads `psnr_table.json`, finds the matching reference per resolution, and
augments each cell's `meta` block with `flip` and `spectrum_json_path`.
Writes per-cell error PNGs at `{rng}__res{R}_spp{N}__error.png`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

os.environ.setdefault("DRJIT_LIBLLVM_PATH", "/opt/homebrew/opt/llvm/lib/libLLVM.dylib")

from randeval.rendering.analysis import (  # noqa: E402
    absolute_error_image, error_to_png_rgb,
    radial_power_spectrum, flip_score,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "results" / "rendering"
SPECTRA_DIR = OUT_DIR / "spectra"
SPECTRA_DIR.mkdir(parents=True, exist_ok=True)


def find_reference(res: int) -> Path | None:
    """Pick the highest-SPP reference for this resolution."""
    candidates = sorted(OUT_DIR.glob(f"reference__res{res}_spp*.npy"))
    if not candidates:
        return None
    # pick the one with the highest SPP
    def spp_of(p: Path) -> int:
        return int(p.stem.split("_spp")[-1])
    return max(candidates, key=spp_of)


def save_png_uint8(img_uint8: np.ndarray, path: Path) -> None:
    import mitsuba as mi
    if "scalar_rgb" not in (mi.variant() or ""):
        mi.set_variant("scalar_rgb")
    bmp = mi.Bitmap(img_uint8)
    bmp.write(str(path))


def main() -> int:
    master_path = OUT_DIR / "psnr_table.json"
    master = json.loads(master_path.read_text())

    refs: dict[int, np.ndarray] = {}
    for res in (16, 32, 64):
        rp = find_reference(res)
        if rp is not None:
            refs[res] = np.load(rp).astype(np.float64)
            print(f"[ref] res{res} <- {rp.name}", flush=True)

    for rng, blob in master.items():
        for res_key in list(blob.keys()):
            if not res_key.startswith("res"):
                continue
            res = int(res_key[3:])
            ref = refs.get(res)
            if ref is None:
                continue
            cell = blob[res_key]
            for spp_key in [k for k in cell if k.startswith("spp")]:
                spp = int(spp_key[3:])
                npy = OUT_DIR / f"{rng}__res{res}_spp{spp}.npy"
                if not npy.exists():
                    continue
                test = np.load(npy).astype(np.float64)
                err = absolute_error_image(test, ref)

                # FLIP
                try:
                    flip_mean, _flip_map = flip_score(test, ref)
                except Exception as e:
                    print(f"[{rng} r{res} spp{spp}] flip failed: {e!r}", flush=True)
                    flip_mean = None

                # spectrum
                freqs, power = radial_power_spectrum(err, num_bins=min(32, max(8, res // 2)))
                spec_path = SPECTRA_DIR / f"{rng}__res{res}_spp{spp}.json"
                spec_path.write_text(json.dumps(
                    {"freq": freqs.tolist(), "power": power.tolist()},
                    indent=2,
                ))

                # error image PNG (use a global vmax per resolution so cells are comparable)
                # we cannot know vmax until we've seen everyone, so use per-cell for now;
                # post-pass can re-render with global if the chapter wants apples-to-apples.
                err_rgb = error_to_png_rgb(err, vmax=None, cmap="magma")
                err_png = OUT_DIR / f"{rng}__res{res}_spp{spp}__error.png"
                save_png_uint8(err_rgb, err_png)

                meta = cell.setdefault("meta", {}).setdefault(spp_key, {})
                meta["flip"] = flip_mean
                meta["spectrum_path"] = str(spec_path.relative_to(REPO_ROOT))
                meta["error_png"] = err_png.name
                print(f"[{rng:30s} r{res} spp{spp}] flip={flip_mean if flip_mean is None else f'{flip_mean:.4f}'}", flush=True)

    master_path.write_text(json.dumps(master, indent=2))
    print(f"\nupdated {master_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
