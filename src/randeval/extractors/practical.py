"""Practical / hardware extractors."""
from __future__ import annotations

import zlib

from typing import cast

import numpy as np
from numpy.typing import NDArray

from .base import Extractor
from ._utils import int_to_bits


class XORExtractor(Extractor):
    """XOR consecutive bit pairs."""

    @property
    def name(self) -> str:
        return "XOR"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if len(data) < 2:
            return np.array([], dtype=np.uint8)
        n = len(data) - len(data) % 2
        pairs = data[:n].reshape(-1, 2)
        return (pairs[:, 0] ^ pairs[:, 1]).astype(np.uint8)


class BlockParityExtractor(Extractor):
    """Divide into k-bit blocks, output parity of each."""

    def __init__(self, block_size: int = 8) -> None:
        self._k = block_size

    @property
    def name(self) -> str:
        return f"BlockParity(k={self._k})"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if len(data) < self._k:
            return np.array([], dtype=np.uint8)
        n = len(data) - len(data) % self._k
        blocks = data[:n].reshape(-1, self._k)
        return cast("NDArray[np.uint8]", (blocks.sum(axis=1) % 2).astype(np.uint8))


class BitDecimationExtractor(Extractor):
    """Take every k-th bit."""

    def __init__(self, step: int = 2) -> None:
        self._step = step

    @property
    def name(self) -> str:
        return f"BitDecimation(k={self._step})"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if len(data) == 0:
            return np.array([], dtype=np.uint8)
        return data[:: self._step].copy()


class WindowedXORExtractor(Extractor):
    """XOR all bits in non-overlapping windows of size w."""

    def __init__(self, window: int = 4) -> None:
        self._w = window

    @property
    def name(self) -> str:
        return f"WindowedXOR(w={self._w})"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if len(data) < self._w:
            return np.array([], dtype=np.uint8)
        n = len(data) - len(data) % self._w
        blocks = data[:n].reshape(-1, self._w)
        return cast("NDArray[np.uint8]", np.bitwise_xor.reduce(blocks, axis=1).astype(np.uint8))


class RepetitionFilter(Extractor):
    """Strip runs of identical bits — keep only the first bit of each run.

    WARNING: This is a preprocessing step, not a standalone extractor. The output
    has structural correlations (run-length information is destroyed) and will fail
    most statistical tests when used alone. Chain with another extractor (e.g.
    VonNeumann or Toeplitz) for proper extraction.
    """

    @property
    def name(self) -> str:
        return "RepetitionFilter"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if len(data) == 0:
            return np.array([], dtype=np.uint8)
        mask = np.ones(len(data), dtype=bool)
        mask[1:] = data[1:] != data[:-1]
        return data[mask].copy()


class ModularReductionExtractor(Extractor):
    """Group bits into k-bit blocks, reduce mod a prime < 2^k.

    Uses rejection sampling to avoid modular bias. However, the output still has
    a residual bit-level bias: uniform integers over [0, modulus) don't produce
    uniform bits when encoded in fixed-width binary (MSB is biased because
    modulus isn't a power of 2). This is inherent to the approach and will cause
    failures on sensitive statistical tests (~60% pass rate typical).
    """

    def __init__(self, block_bits: int = 8, modulus: int = 251) -> None:
        self._k = block_bits
        self._mod = modulus
        self._out_bits = max(1, (modulus - 1).bit_length())
        # rejection threshold: largest multiple of modulus <= 2^k
        self._threshold = (1 << block_bits) // modulus * modulus

    @property
    def name(self) -> str:
        return f"ModReduction(k={self._k},m={self._mod})"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if len(data) < self._k:
            return np.array([], dtype=np.uint8)
        n = len(data) - len(data) % self._k
        blocks = data[:n].reshape(-1, self._k)
        powers = (2 ** np.arange(self._k - 1, -1, -1))
        vals = (blocks * powers).sum(axis=1)
        # reject values that would cause modular bias
        mask = vals < self._threshold
        reduced = vals[mask] % self._mod
        output: list[int] = []
        for v in reduced:
            bits = int_to_bits(int(v), self._out_bits)
            output.extend(bits)
        return np.array(output, dtype=np.uint8)


class CRCExtractor(Extractor):
    """CRC32 as a fast linear hash."""

    @property
    def name(self) -> str:
        return "CRC32"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        packed = np.packbits(data).tobytes()
        crc = zlib.crc32(packed) & 0xFFFFFFFF
        return int_to_bits(crc, 32)


class SubsamplingExtractor(Extractor):
    """Seeded random index selection."""

    def __init__(self, output_bits: int = 64, seed: int = 0) -> None:
        self._m = output_bits
        self._seed = seed

    @property
    def name(self) -> str:
        return f"Subsampling(m={self._m})"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        n = len(data)
        m = min(self._m, n)
        rng = np.random.default_rng(self._seed)
        indices = rng.choice(n, size=m, replace=False)
        indices.sort()
        return data[indices].copy()


class Condenser(Extractor):
    """XOR-based condenser — XOR groups of rate bits down to 1."""

    def __init__(self, rate: int = 2) -> None:
        self._inner = WindowedXORExtractor(window=rate)
        self._rate = rate

    @property
    def name(self) -> str:
        return f"Condenser(r={self._rate})"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        return self._inner.extract(data)
