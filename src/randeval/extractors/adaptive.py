"""Adaptive extractors: MinEntropy, Fuzzy, ArithmeticCoding."""
from __future__ import annotations

import hashlib
import math

import numpy as np
from numpy.typing import NDArray

from .base import Extractor
from ._utils import random_binary_seed


class MinEntropyExtractor(Extractor):
    """Estimates min-entropy then extracts that many bits via Toeplitz hashing."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    @property
    def name(self) -> str:
        return "MinEntropyExtractor"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if len(data) < 8:
            return np.array([], dtype=np.uint8)
        p1 = float(data.mean())
        p_max = max(p1, 1 - p1)
        if p_max >= 1.0:
            return np.array([], dtype=np.uint8)
        h_min = -math.log2(p_max)
        m = max(1, int(h_min * len(data) * 0.9))
        m = min(m, len(data) - 1, 10_000)  # cap output — Toeplitz is O(m*n)
        from .hashing import ToeplitzHash
        return ToeplitzHash(output_bits=m, seed=self._seed).extract(data)


class FuzzyExtractor(Extractor):
    """Fuzzy extractor — Gen/Rep paradigm for noisy sources (Dodis et al. 2008).

    NOTE: extract() always outputs 256 bits (SHA-256 hash of the lock value),
    regardless of input size. This is too short for most statistical tests and
    will score poorly when evaluated standalone. The value of a fuzzy extractor
    is reproducibility from noisy re-readings, not statistical quality of a
    single extraction. Use generate()/reproduce() for the intended workflow.
    """

    def __init__(self, seed: int = 0, error_tolerance: float = 0.05) -> None:
        self._seed = seed
        self._tol = error_tolerance

    @property
    def name(self) -> str:
        return "FuzzyExtractor"

    def generate(self, data: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Generate a key and helper string from an initial reading."""
        rng = np.random.default_rng(self._seed)
        lock = random_binary_seed(len(data), rng)
        helper = (data ^ lock).astype(np.uint8)
        packed = np.packbits(lock).tobytes()
        digest = hashlib.sha256(packed).digest()
        key = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))
        return key, helper

    def reproduce(self, noisy: NDArray[np.uint8], helper: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Recover the key from a noisy re-reading and the helper string."""
        lock_candidate = (noisy ^ helper).astype(np.uint8)
        packed = np.packbits(lock_candidate).tobytes()
        digest = hashlib.sha256(packed).digest()
        return np.unpackbits(np.frombuffer(digest, dtype=np.uint8))

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        key, _ = self.generate(data)
        return key


class ArithmeticCodingExtractor(Extractor):
    """Near-optimal extraction from biased source via arithmetic coding."""

    def __init__(self, bias: float = 0.5) -> None:
        if bias <= 0.0 or bias >= 1.0:
            raise ValueError(f"bias must be in (0, 1), got {bias}")
        self._bias = bias

    @property
    def name(self) -> str:
        return f"ArithmeticCoding(p={self._bias:.2f})"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if len(data) == 0:
            return np.array([], dtype=np.uint8)
        p = self._bias
        q = 1 - p
        lo = 0.0
        hi = 1.0
        output: list[int] = []
        for bit in data:
            mid_point = lo + (hi - lo) * q
            if bit == 0:
                hi = mid_point
            else:
                lo = mid_point
            while True:
                if hi <= 0.5:
                    output.append(0)
                    lo *= 2
                    hi *= 2
                elif lo >= 0.5:
                    output.append(1)
                    lo = (lo - 0.5) * 2
                    hi = (hi - 0.5) * 2
                else:
                    break
        return np.array(output, dtype=np.uint8) if output else np.array([], dtype=np.uint8)
