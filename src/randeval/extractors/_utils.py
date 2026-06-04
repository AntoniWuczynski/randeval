"""Shared utilities for extractor implementations."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def gf2_matmul(mat: NDArray[np.uint8], vec: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Multiply a binary matrix by a binary vector over GF(2)."""
    return (mat @ vec) % 2


def random_binary_seed(
    n: int, rng: np.random.Generator | None = None
) -> NDArray[np.uint8]:
    """Generate n random bits as a uint8 array."""
    rng = rng or np.random.default_rng()
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def estimate_bias(data: NDArray[np.uint8]) -> float:
    """Estimate the probability of a 1-bit."""
    return float(data.mean())


def bits_to_int(bits: NDArray[np.uint8]) -> int:
    """Convert a bit array (MSB first) to an integer."""
    return int.from_bytes(np.packbits(np.pad(bits, (0, (-len(bits)) % 8))).tobytes(), "big") >> ((-len(bits)) % 8)


def int_to_bits(val: int, width: int) -> NDArray[np.uint8]:
    """Convert an integer to a bit array of given width (MSB first)."""
    nbytes = (width + 7) // 8
    raw = val.to_bytes(nbytes, "big")
    all_bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return all_bits[len(all_bits) - width:]


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _has_primitive_root_2(p: int) -> bool:
    if p == 2:
        return True
    phi = p - 1
    factors = set()
    n = phi
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return all(pow(2, phi // q, p) != 1 for q in factors)


def next_prime_with_primitive_root_2(start: int) -> int:
    """Find the smallest prime >= start where 2 is a primitive root."""
    n = start if start >= 2 else 2
    while True:
        if _is_prime(n) and _has_primitive_root_2(n):
            return n
        n += 1


def require_package(package: str, pip_extra: str | None = None) -> None:
    """Check that a package is importable, raise with install hint if not."""
    try:
        __import__(package)
    except ImportError as exc:
        hint = f"pip install randeval[{pip_extra}]" if pip_extra else f"pip install {package}"
        raise ImportError(f"'{package}' is required. Install with: {hint}") from exc
