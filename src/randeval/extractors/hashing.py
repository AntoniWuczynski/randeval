"""Universal hashing extractors: Toeplitz, Linear, InnerProduct, LHL, Polynomial."""
from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from .base import Extractor
from ._utils import gf2_matmul, random_binary_seed, int_to_bits, bits_to_int


class ToeplitzHash(Extractor):
    """Toeplitz matrix hashing — universal hash family for randomness extraction."""

    def __init__(self, output_bits: int = 64, seed: int = 0) -> None:
        self._m = output_bits
        self._seed = seed

    @property
    def name(self) -> str:
        return f"ToeplitzHash(m={self._m})"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        n = len(data)
        m = min(self._m, n)
        rng = np.random.default_rng(self._seed)
        seed_vec = rng.integers(0, 2, size=n + m - 1, dtype=np.uint8)
        # row-by-row dot product avoids materialising m×n matrix
        out = np.empty(m, dtype=np.uint8)
        for i in range(m):
            out[i] = np.dot(seed_vec[i:i + n], data) % 2
        return out


class LinearHash(Extractor):
    """Random binary matrix over GF(2)."""

    def __init__(self, output_bits: int = 32, seed: int = 0) -> None:
        self._m = output_bits
        self._seed = seed

    @property
    def name(self) -> str:
        return f"LinearHash(m={self._m})"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        n = len(data)
        m = min(self._m, n)
        rng = np.random.default_rng(self._seed)
        # row-by-row to avoid m×n matrix allocation
        out = np.empty(m, dtype=np.uint8)
        for i in range(m):
            row = rng.integers(0, 2, size=n, dtype=np.uint8)
            out[i] = np.dot(row, data) % 2
        return out


class InnerProductExtractor(Extractor):
    """GF(2) inner product two-source extractor (Chor & Goldreich 1988)."""

    def __init__(self, block_size: int = 8) -> None:
        self._block = block_size

    @property
    def name(self) -> str:
        return "InnerProduct"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if len(data) < 2:
            return np.array([], dtype=np.uint8)
        half = len(data) // 2
        a, b = data[:half], data[half : 2 * half]
        bs = self._block
        n_blocks = half // bs
        if n_blocks == 0:
            return np.array([int(np.sum(a * b) % 2)], dtype=np.uint8)
        a_blocks = a[: n_blocks * bs].reshape(n_blocks, bs)
        b_blocks = b[: n_blocks * bs].reshape(n_blocks, bs)
        return cast("NDArray[np.uint8]", ((a_blocks * b_blocks).sum(axis=1) % 2).astype(np.uint8))


class LHLHash(Extractor):
    """Polynomial universal hash per the Leftover Hash Lemma."""

    def __init__(self, output_bits: int = 32, seed: int = 0) -> None:
        self._m = output_bits
        self._seed = seed

    @property
    def name(self) -> str:
        return f"LHLHash(m={self._m})"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        # Toeplitz-based universal hash, row-by-row to avoid OOM
        n = len(data)
        m = min(self._m, n)
        rng = np.random.default_rng(self._seed)
        seed_bits = rng.integers(0, 2, size=n + m - 1, dtype=np.uint8)
        out = np.empty(m, dtype=np.uint8)
        for i in range(m):
            out[i] = np.dot(seed_bits[i:i + n], data) % 2
        return out


class PolynomialHash(Extractor):
    """Polynomial evaluation over GF(2^n) — algebraic universal hash family."""

    def __init__(self, output_bits: int = 32, seed: int = 0) -> None:
        self._m = output_bits
        self._seed = seed

    @property
    def name(self) -> str:
        return f"PolynomialHash(m={self._m})"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        n = len(data)
        m = min(self._m, n)
        rng = np.random.default_rng(self._seed)
        block_count = n // m if m > 0 else 0
        if block_count == 0:
            return random_binary_seed(m, rng)
        coeffs = data[: block_count * m].reshape(block_count, m)
        point = random_binary_seed(m, rng)
        # Horner's method over GF(2^m)
        result = coeffs[-1].copy()
        for i in range(block_count - 2, -1, -1):
            result = _gf2m_mul(result, point, m)
            result = result ^ coeffs[i]
        return cast("NDArray[np.uint8]", result.astype(np.uint8))


def _gf2m_mul(a: NDArray[np.uint8], b: NDArray[np.uint8], m: int) -> NDArray[np.uint8]:
    """Carry-less polynomial multiplication in GF(2^m), reduced mod x^m + x + 1."""
    a_int = bits_to_int(a)
    b_int = bits_to_int(b)
    # carry-less multiply
    product = 0
    for i in range(m):
        if (b_int >> i) & 1:
            product ^= a_int << i
    # reduce mod irreducible polynomial x^m + x + 1
    modpoly = (1 << m) | 0b11  # x^m + x + 1
    for i in range(2 * m - 2, m - 1, -1):
        if (product >> i) & 1:
            product ^= modpoly << (i - m)
    return int_to_bits(product & ((1 << m) - 1), m)
