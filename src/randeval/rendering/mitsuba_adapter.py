"""Mitsuba 3 sampler that pulls floats from a randeval bit stream.

Registers under sampler-name "randeval". Use via load_dict:
    {"type": "randeval"}
…after calling install_randeval_sampler(bits).
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .bitstream import BitFloatStream


_BITS_HOLDER: dict[str, BitFloatStream] = {}


def _ensure_llvm_env() -> None:
    """drjit on macOS won't init without the LLVM dylib path. Try common locations."""
    if os.environ.get("DRJIT_LIBLLVM_PATH"):
        return
    for cand in (
        "/opt/homebrew/opt/llvm/lib/libLLVM.dylib",
        "/usr/local/opt/llvm/lib/libLLVM.dylib",
    ):
        if os.path.exists(cand):
            os.environ["DRJIT_LIBLLVM_PATH"] = cand
            return


def install_randeval_sampler(bits: NDArray[np.uint8], bits_per_value: int = 32) -> BitFloatStream:
    """Register the 'randeval' sampler type with Mitsuba and bind a bit stream.

    Returns the BitFloatStream so the caller can read .wraps after rendering.
    """
    _ensure_llvm_env()
    import mitsuba as mi  # local import: variant must be set first

    stream = BitFloatStream(bits, bits_per_value=bits_per_value)
    _BITS_HOLDER["stream"] = stream

    class RandevalSampler(mi.Sampler):  # type: ignore[misc]
        def __init__(self, props: Any) -> None:
            super().__init__(props)

        def next_1d(self, active: bool = True) -> float:
            return _BITS_HOLDER["stream"].next_float()

        def next_2d(self, active: bool = True) -> Any:
            u = _BITS_HOLDER["stream"].next_float()
            v = _BITS_HOLDER["stream"].next_float()
            return mi.Point2f(u, v)

        def seed(self, seed_value: int, wavefront_size: int = 1) -> None:
            # bit stream is the seed; ignore Mitsuba's seed
            pass

        def fork(self) -> Any:
            return RandevalSampler(mi.Properties())

        def clone(self) -> Any:
            return RandevalSampler(mi.Properties())

        def advance(self) -> None:
            pass

        def to_string(self) -> str:
            s = _BITS_HOLDER.get("stream")
            pos = s.position if s else -1
            return f"RandevalSampler[pos={pos}]"

    mi.register_sampler("randeval", lambda props: RandevalSampler(props))
    return stream


def _build_scene_dict(scene_name: str) -> dict[str, Any]:
    """Return a Mitsuba dict scene by name."""
    import mitsuba as mi
    sd: dict[str, Any] = mi.cornell_box()
    if scene_name == "cornell":
        return sd
    if scene_name == "cornell_glass":
        # replace the small box with a glass sphere
        sd.pop("small-box", None)
        sd["glass-sphere"] = {
            "type": "sphere",
            "center": [0.34, -0.65, 0.38],
            "radius": 0.32,
            "bsdf": {
                "type": "dielectric",
                "int_ior": "bk7",
                "ext_ior": "air",
            },
        }
        return sd
    raise ValueError(f"unknown scene: {scene_name}")


def render_cornell_mitsuba(
    bits: NDArray[np.uint8] | None,
    width: int,
    height: int,
    spp: int,
    *,
    bits_per_value: int = 32,
    scene: str = "cornell",
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Render a Mitsuba dict scene, optionally driven by a randeval bit stream.

    If bits is None, uses the default 'independent' sampler (PCG-based).
    """
    _ensure_llvm_env()
    import mitsuba as mi
    if "scalar_rgb" not in (mi.variant() or ""):
        mi.set_variant("scalar_rgb")

    scene_d = _build_scene_dict(scene)
    scene_d["sensor"]["film"]["width"] = width
    scene_d["sensor"]["film"]["height"] = height

    stream: BitFloatStream | None = None
    if bits is not None:
        stream = install_randeval_sampler(bits, bits_per_value=bits_per_value)
        scene_d["sensor"]["sampler"] = {"type": "randeval"}
    else:
        scene_d["sensor"]["sampler"] = {"type": "independent"}

    sc = mi.load_dict(scene_d)
    integ = sc.integrator()
    sensor = sc.sensors()[0]
    img = integ.render(scene=sc, sensor=sensor, seed=0, spp=spp,
                       develop=True, evaluate=False)
    arr = np.array(img, dtype=np.float32)
    info: dict[str, Any] = {"width": width, "height": height, "spp": spp, "scene": scene}
    if stream is not None:
        info["floats_consumed"] = stream.floats_consumed()
        info["wraps"] = stream.wraps
    return arr, info
