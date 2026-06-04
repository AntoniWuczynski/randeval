"""Debiasing extractors: Peres, Elias, AMLS."""
from __future__ import annotations

import math
from math import comb

import numpy as np
from numpy.typing import NDArray

from .base import Extractor


class PeresExtractor(Extractor):
    """Iterated Von Neumann — recursively extracts from discarded pairs (Peres 1992)."""

    @property
    def name(self) -> str:
        return "Peres"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        bits = list(data)
        return np.array(self._peres_recursive(bits), dtype=np.uint8)

    def _peres_recursive(self, bits: list[int]) -> list[int]:
        if len(bits) < 2:
            return []
        n = len(bits) - len(bits) % 2
        output: list[int] = []
        discarded_xor: list[int] = []
        for i in range(0, n, 2):
            a, b = bits[i], bits[i + 1]
            if a != b:
                output.append(a)
            discarded_xor.append(a ^ b)
        output.extend(self._peres_recursive(discarded_xor))
        return output


class EliasExtractor(Extractor):
    """Elias debiasing for known bias p (Elias 1972).

    Groups bits into blocks, maps blocks to output bits using
    the probability ordering.
    """

    def __init__(self, bias: float) -> None:
        if bias <= 0.0 or bias >= 1.0:
            raise ValueError(f"bias must be in (0, 1), got {bias}")
        self._bias = bias

    @property
    def name(self) -> str:
        return f"Elias(p={self._bias:.2f})"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if len(data) == 0:
            return np.array([], dtype=np.uint8)
        p = self._bias
        h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        block_size = max(2, min(20, int(math.ceil(1.0 / h)) + 1))
        return self._elias_extract(data, block_size)

    def _elias_extract(self, data: NDArray[np.uint8], block_size: int) -> NDArray[np.uint8]:
        n_blocks = len(data) // block_size
        if n_blocks == 0:
            return np.array([], dtype=np.uint8)
        blocks = data[: n_blocks * block_size].reshape(n_blocks, block_size)
        output: list[int] = []
        for block in blocks:
            ones = int(block.sum())
            n_same_weight = comb(block_size, ones)
            if n_same_weight < 2:
                continue
            n_bits = int(math.log2(n_same_weight))
            usable = 1 << n_bits  # 2^n_bits
            rank = self._rank_in_weight_class(block, ones)
            # reject ranks >= 2^n_bits to avoid bias
            if rank >= usable:
                continue
            for i in range(n_bits - 1, -1, -1):
                output.append((rank >> i) & 1)
        return np.array(output, dtype=np.uint8) if output else np.array([], dtype=np.uint8)

    @staticmethod
    def _rank_in_weight_class(block: NDArray[np.uint8], weight: int) -> int:
        rank = 0
        w = weight
        n = len(block)
        for i, bit in enumerate(block):
            remaining = n - i - 1
            if bit == 1:
                rank += comb(remaining, w)
                w -= 1
        return rank


class AMLSExtractor(Extractor):
    """Advanced Multi-Level Strategy — optimal extraction from biased coins (Pae & Loui 2005)."""

    def __init__(self, max_depth: int = 20) -> None:
        self._max_depth = max_depth

    @property
    def name(self) -> str:
        return "AMLS"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if len(data) < 2:
            return np.array([], dtype=np.uint8)
        output = self._amls(data.tolist(), self._max_depth)
        return np.array(output, dtype=np.uint8) if output else np.array([], dtype=np.uint8)

    def _amls(self, bits: list[int], depth: int) -> list[int]:
        if len(bits) < 2 or depth <= 0:
            return []
        n = len(bits) - len(bits) % 2
        vn_output: list[int] = []
        xor_stream: list[int] = []
        agree_stream: list[int] = []
        for i in range(0, n, 2):
            a, b = bits[i], bits[i + 1]
            if a != b:
                vn_output.append(a)
                xor_stream.append(1)
            else:
                xor_stream.append(0)
                agree_stream.append(a)
        result = list(vn_output)
        result.extend(self._amls(xor_stream, depth - 1))
        result.extend(self._amls(agree_stream, depth - 1))
        return result
