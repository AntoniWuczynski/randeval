"""Post-render analysis: error images, FLIP, radial Fourier spectrum.

Operates on the .npy files written by render_poc.py — does not re-render.
"""
from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import NDArray


def absolute_error_image(test: NDArray[np.float64], ref: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-pixel L2 magnitude of the (test - ref) RGB difference, scalar HxW."""
    diff = test - ref
    return cast("NDArray[np.float64]", np.sqrt(np.sum(diff * diff, axis=-1)))


def error_to_png_rgb(
    err: NDArray[np.float64],
    *,
    vmax: float | None = None,
    cmap: str = "magma",
) -> NDArray[np.uint8]:
    """Map a scalar error image to an RGB uint8 PNG-ready array."""
    if vmax is None:
        vmax = float(err.max()) or 1.0
    norm = np.clip(err / vmax, 0.0, 1.0)
    # cheap inline magma-ish ramp (avoid matplotlib dependency)
    if cmap == "magma":
        # 5-stop linear interpolation in RGB
        stops = np.array([
            [0.0, 0.0, 0.05],
            [0.30, 0.05, 0.40],
            [0.65, 0.20, 0.55],
            [0.95, 0.55, 0.40],
            [1.00, 0.95, 0.65],
        ])
        idx = norm * (len(stops) - 1)
        lo = np.clip(np.floor(idx).astype(int), 0, len(stops) - 1)
        hi = np.clip(lo + 1, 0, len(stops) - 1)
        t = (idx - lo)[..., None]
        rgb = stops[lo] * (1 - t) + stops[hi] * t
    else:
        rgb = np.repeat(norm[..., None], 3, axis=-1)
    return cast("NDArray[np.uint8]", (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))


def radial_power_spectrum(err: NDArray[np.float64], *, num_bins: int = 32) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """1D radial average of the 2D Fourier power spectrum of an error image.

    Returns (frequencies in cycles/pixel, mean power per bin).
    """
    h, w = err.shape
    f = np.fft.fft2(err - err.mean())
    p = np.abs(np.fft.fftshift(f)) ** 2

    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = r.max()

    bins = np.linspace(0, r_max, num_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means = np.empty(num_bins, dtype=np.float64)
    for i in range(num_bins):
        m = (r >= bins[i]) & (r < bins[i + 1])
        means[i] = float(p[m].mean()) if m.any() else 0.0

    nyquist = 0.5  # cycles/pixel
    freqs = centers / r_max * nyquist
    return freqs, means


def flip_score(test: NDArray[np.float64], ref: NDArray[np.float64]) -> tuple[float, NDArray[np.float32]]:
    """FLIP perceptual error. Inputs are linear HDR; clipped to LDR for FLIP."""
    import flip_evaluator as fe
    ref_clip = np.clip(ref.astype(np.float32), 0.0, 1.0)
    test_clip = np.clip(test.astype(np.float32), 0.0, 1.0)
    err_map, mean_err, _ = fe.evaluate(
        ref_clip, test_clip, "LDR",
        inputsRGB=True, applyMagma=False, computeMeanError=True,
    )
    return float(mean_err), err_map.astype(np.float32)
