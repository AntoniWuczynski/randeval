"""Reference inputs and known-answer values for validating the statistical tests.

The NIST SP 800-22 Rev 1a document (§2) ships worked examples with published
P-values. We anchor on two reproducible inputs:

  * the 100-bit example sequence (first 100 bits of the binary expansion of pi),
    used throughout §2.x.8;
  * NIST's own ``data.e`` / ``data.pi`` reference files (vendored under
    ``fixtures/``), whose first 1,000,000 bits reproduce the Serial worked
    example exactly (#1s = 500029, #0s = 499971).

A few of NIST's *printed* example P-values are internally inconsistent (the
arithmetic does not follow from their own stated intermediates). Those are
flagged below as errata and cross-validated against an independent
implementation (``nistrng``) instead. See VALIDATION.md for the full write-up.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_FIX = Path(__file__).parent / "fixtures"

# First 100 bits of the binary expansion of pi — the SP 800-22 §2 example
# sequence. Verified byte-identical to data.pi[:100].
PI_100 = (
    "1100100100001111110110101010001000100001011010001100"
    "001000110100110001001100011001100010100010111000"
)


def bits(text: str) -> NDArray[np.uint8]:
    """Turn a '0'/'1' string into a uint8 bit array."""
    arr = np.frombuffer(text.encode(), dtype=np.uint8) - ord("0")
    return arr.astype(np.uint8)


def _load(name: str) -> NDArray[np.uint8]:
    raw = (_FIX / name).read_text()
    return bits(raw.strip())


def pi_100() -> NDArray[np.uint8]:
    """The 100-bit pi example sequence as a bit array."""
    return bits(PI_100)


def e_bits(n: int | None = None) -> NDArray[np.uint8]:
    """First ``n`` bits of NIST's data.e (full file if n is None)."""
    b = _load("nist_e_bits.txt")
    return b if n is None else b[:n]


def pi_bits(n: int | None = None) -> NDArray[np.uint8]:
    """First ``n`` bits of NIST's data.pi (full file if n is None)."""
    b = _load("nist_pi_bits.txt")
    return b if n is None else b[:n]
