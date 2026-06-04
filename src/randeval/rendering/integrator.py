"""Pure-Python direct-illumination integrator for a Cornell-box scene.

Used as a fallback when Mitsuba's Python sampler trampoline does not pan out,
or simply to side-step LLVM dependencies. Diffuse-only, area-light sampling.

Coordinates: box from (0,0,0) to (1,1,1). Camera looks down +z.
The geometry is hard-coded — this is not a general renderer.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .bitstream import BitFloatStream


# ─────── geometry ────────────────────────────────────────────────────────────

@dataclass
class Quad:
    """Axis-aligned quad. `axis` is the constant axis (0=x,1=y,2=z),
    `value` is the coordinate on that axis, the two-tuples are (min,max)
    on the other axes (in the natural order).
    """
    axis: int
    value: float
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    normal: tuple[float, float, float]
    albedo: tuple[float, float, float]
    emission: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def area(self) -> float:
        return (self.u_range[1] - self.u_range[0]) * (self.v_range[1] - self.v_range[0])

    def sample_point(self, u: float, v: float) -> NDArray[np.float64]:
        """Uniform point on the quad given (u,v) in [0,1)^2."""
        p = np.zeros(3, dtype=np.float64)
        p[self.axis] = self.value
        oa, ob = [a for a in range(3) if a != self.axis]
        p[oa] = self.u_range[0] + u * (self.u_range[1] - self.u_range[0])
        p[ob] = self.v_range[0] + v * (self.v_range[1] - self.v_range[0])
        return p


def cornell_quads() -> tuple[list[Quad], Quad]:
    """Return (all_quads, light_quad). The light is duplicated in all_quads
    so that camera rays can hit it directly (visible emitter).
    """
    white = (0.73, 0.73, 0.73)
    red = (0.65, 0.05, 0.05)
    green = (0.12, 0.45, 0.15)
    Lcolor = (15.0, 15.0, 15.0)

    floor = Quad(1, 0.0, (0.0, 1.0), (0.0, 1.0), (0, 1, 0), white)
    ceiling = Quad(1, 1.0, (0.0, 1.0), (0.0, 1.0), (0, -1, 0), white)
    back = Quad(2, 1.0, (0.0, 1.0), (0.0, 1.0), (0, 0, -1), white)
    left = Quad(0, 0.0, (0.0, 1.0), (0.0, 1.0), (1, 0, 0), red)
    right = Quad(0, 1.0, (0.0, 1.0), (0.0, 1.0), (-1, 0, 0), green)

    # ceiling-mounted square light, slightly inset, just below ceiling
    light = Quad(
        axis=1,
        value=0.999,
        u_range=(0.35, 0.65),
        v_range=(0.35, 0.65),
        normal=(0, -1, 0),
        albedo=(0.0, 0.0, 0.0),
        emission=Lcolor,
    )

    return [floor, ceiling, back, left, right, light], light


def intersect_quad(orig: NDArray[np.float64], dir_: NDArray[np.float64], q: Quad) -> float:
    """Return ray param t > 0 of intersection, or +inf if none."""
    d_axis = dir_[q.axis]
    if abs(d_axis) < 1e-12:
        return math.inf
    t = (q.value - orig[q.axis]) / d_axis
    if t <= 1e-4:
        return math.inf
    oa, ob = [a for a in range(3) if a != q.axis]
    pu = orig[oa] + t * dir_[oa]
    pv = orig[ob] + t * dir_[ob]
    if not (q.u_range[0] <= pu <= q.u_range[1]):
        return math.inf
    if not (q.v_range[0] <= pv <= q.v_range[1]):
        return math.inf
    return float(t)


def trace(orig: NDArray[np.float64], dir_: NDArray[np.float64], quads: list[Quad]) -> tuple[float, Quad | None]:
    best_t = math.inf
    best_q: Quad | None = None
    for q in quads:
        t = intersect_quad(orig, dir_, q)
        if t < best_t:
            best_t = t
            best_q = q
    return best_t, best_q


# ─────── shading ─────────────────────────────────────────────────────────────

def shade_direct(
    hit_p: NDArray[np.float64],
    hit_q: Quad,
    quads: list[Quad],
    light: Quad,
    stream: BitFloatStream,
) -> NDArray[np.float64]:
    """Direct illumination via area-light sampling.

    BRDF is Lambertian (albedo / pi). Visibility tested with a shadow ray.
    """
    if hit_q.emission != (0.0, 0.0, 0.0):
        return np.array(hit_q.emission, dtype=np.float64)

    u = stream.next_float()
    v = stream.next_float()
    lp = light.sample_point(u, v)
    to_l = lp - hit_p
    dist2 = float(np.dot(to_l, to_l))
    dist = math.sqrt(dist2)
    wi = to_l / dist

    n = np.array(hit_q.normal, dtype=np.float64)
    nl = np.array(light.normal, dtype=np.float64)
    cos_surf = float(np.dot(n, wi))
    cos_light = float(np.dot(nl, -wi))
    if cos_surf <= 0.0 or cos_light <= 0.0:
        return np.zeros(3)

    # shadow test
    shadow_orig = hit_p + 1e-4 * n
    t_block, q_block = trace(shadow_orig, wi, quads)
    if q_block is not light or t_block + 1e-3 < dist:
        return np.zeros(3)

    albedo = np.array(hit_q.albedo, dtype=np.float64)
    Le = np.array(light.emission, dtype=np.float64)
    pdf_area = 1.0 / light.area()
    geom = (cos_surf * cos_light) / dist2
    return albedo / math.pi * Le * geom / pdf_area


# ─────── camera + render loop ────────────────────────────────────────────────

def make_camera_ray(px: float, py: float, width: int, height: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pinhole camera at (0.5, 0.5, -1.5) looking towards +z."""
    aspect = width / height
    fov_y = 40.0 * math.pi / 180.0
    h = math.tan(fov_y / 2.0)
    w = h * aspect
    # NDC in [-1, 1]
    ndc_x = (px / width) * 2.0 - 1.0
    ndc_y = 1.0 - (py / height) * 2.0
    dir_ = np.array([ndc_x * w, ndc_y * h, 1.0], dtype=np.float64)
    dir_ /= np.linalg.norm(dir_)
    orig = np.array([0.5, 0.5, -1.4], dtype=np.float64)
    return orig, dir_


def render_cornell(
    bits: NDArray[np.uint8] | None,
    width: int,
    height: int,
    spp: int,
    *,
    bits_per_value: int = 32,
    seed: int = 0,
) -> tuple[NDArray[np.float64], dict[str, int]]:
    """Render the Cornell box. If bits is None, fall back to numpy default RNG.

    Returns (image HxWx3 float64 in linear space, info dict).
    """
    quads, light = cornell_quads()

    draw: Callable[[], float]
    wraps_getter: Callable[[], int]
    consumed_getter: Callable[[], int]
    if bits is not None:
        stream = BitFloatStream(bits, bits_per_value=bits_per_value)
        def draw() -> float:
            return stream.next_float()
        wraps_getter = lambda: stream.wraps
        consumed_getter = lambda: stream.floats_consumed()
    else:
        rng = np.random.default_rng(seed)
        def draw() -> float:
            return float(rng.random())
        wraps_getter = lambda: 0
        consumed_getter = lambda: -1

    img = np.zeros((height, width, 3), dtype=np.float64)
    inv_spp = 1.0 / spp
    for y in range(height):
        for x in range(width):
            acc = np.zeros(3, dtype=np.float64)
            for _ in range(spp):
                jx = draw()
                jy = draw()
                orig, d = make_camera_ray(x + jx, y + jy, width, height)
                t, q = trace(orig, d, quads)
                if q is None:
                    continue
                # need a stream-backed shader; pass a tiny stream view
                # by using closure draws — build a temporary BitFloatStream-like
                hit_p = orig + t * d
                # pull two more numbers for the area-light sample directly
                lu = draw()
                lv = draw()
                lp = light.sample_point(lu, lv)
                to_l = lp - hit_p
                dist2 = float(np.dot(to_l, to_l))
                dist = math.sqrt(dist2)
                wi = to_l / dist

                if q.emission != (0.0, 0.0, 0.0):
                    acc += np.array(q.emission, dtype=np.float64)
                    continue

                n = np.array(q.normal, dtype=np.float64)
                nl = np.array(light.normal, dtype=np.float64)
                cos_surf = float(np.dot(n, wi))
                cos_light = float(np.dot(nl, -wi))
                if cos_surf <= 0.0 or cos_light <= 0.0:
                    continue
                shadow_orig = hit_p + 1e-4 * n
                t_block, q_block = trace(shadow_orig, wi, quads)
                if q_block is not light or t_block + 1e-3 < dist:
                    continue
                albedo = np.array(q.albedo, dtype=np.float64)
                Le = np.array(light.emission, dtype=np.float64)
                pdf_area = 1.0 / light.area()
                geom = (cos_surf * cos_light) / dist2
                acc += albedo / math.pi * Le * geom / pdf_area
            img[y, x] = acc * inv_spp

    info = {
        "wraps": wraps_getter(),
        "floats_consumed": consumed_getter(),
        "width": width, "height": height, "spp": spp,
    }
    return img, info


def to_srgb_uint8(img: NDArray[np.float64]) -> NDArray[np.uint8]:
    """Linear -> sRGB tonemap, clip to [0, 255] uint8."""
    x = np.clip(img, 0.0, None)
    # simple gamma 1/2.2; not full sRGB but visually fine
    g = np.power(x / (x.max() + 1e-12) if x.max() > 1.0 else x, 1.0 / 2.2)
    g = np.clip(g, 0.0, 1.0)
    return (g * 255.0 + 0.5).astype(np.uint8)


def psnr(ref: NDArray[np.float64], test: NDArray[np.float64], peak: float = 1.0) -> float:
    """PSNR between two linear images, peak=1.0 by convention for tonemapped HDR."""
    diff = ref - test
    mse = float(np.mean(diff * diff))
    if mse <= 1e-20:
        return float("inf")
    return 10.0 * math.log10(peak * peak / mse)
