"""Shared utilities for statistical test implementations."""
from __future__ import annotations

from math import erfc, sqrt, lgamma, log, exp
from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats
from scipy.special import gammaincc

from .base import Verdict


def bits_to_blocks(data: NDArray[np.uint8], block_size: int) -> NDArray[np.int64]:
    """Pack consecutive bits into integer blocks, truncating trailing bits.

    Args:
        data: 1-D array of 0s and 1s.
        block_size: Number of bits per block.

    Returns:
        NDArray[np.int64]: Array of integer values, length n // block_size.
    """
    n = len(data) - (len(data) % block_size)
    trimmed = data[:n].reshape(-1, block_size)
    powers = (2 ** np.arange(block_size - 1, -1, -1)).astype(np.int64)
    return cast("NDArray[np.int64]", (trimmed * powers).sum(axis=1))


def bits_to_bytes(data: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Pack bits into bytes, zero-padding if length isn't a multiple of 8.

    Args:
        data: 1-D array of 0s and 1s.

    Returns:
        NDArray[np.uint8]: Packed byte array, length ceil(n / 8).
    """
    padded = np.pad(data, (0, (-len(data)) % 8), constant_values=0)
    return np.packbits(padded)


def bits_to_floats(data: NDArray[np.uint8], bits_per_value: int = 32) -> NDArray[np.float64]:
    """Convert consecutive bit blocks to floats in [0, 1).

    Args:
        data: 1-D array of 0s and 1s.
        bits_per_value: Number of bits per float value.

    Returns:
        NDArray[np.float64]: Array of floats in [0, 1), length n // bits_per_value.
    """
    blocks = bits_to_blocks(data, bits_per_value)
    return cast("NDArray[np.float64]", blocks.astype(np.float64) / (2 ** bits_per_value))


def proportion_ones(data: NDArray[np.uint8]) -> float:
    """Fraction of 1-bits in the array.

    Args:
        data: 1-D array of 0s and 1s.

    Returns:
        float: Proportion of ones, between 0.0 and 1.0.
    """
    return float(data.sum()) / len(data)


def p_from_z(z: float) -> float:
    """Two-tailed p-value from a standard normal z-score.

    Args:
        z: Standard normal z-score.

    Returns:
        float: Two-tailed p-value via erfc.
    """
    return erfc(abs(z) / sqrt(2.0))


def p_from_chi2(statistic: float, df: int) -> float:
    """Upper-tail p-value from a chi-squared distribution.

    Args:
        statistic: Chi-squared test statistic.
        df: Degrees of freedom.

    Returns:
        float: Upper-tail p-value via the regularised incomplete gamma function.
    """
    return float(gammaincc(df / 2.0, statistic / 2.0))


def verdict(p_value: float, alpha: float = 0.01) -> Verdict:
    """Decide PASS or FAIL from a p-value and significance level.

    Args:
        p_value: Observed p-value.
        alpha: Significance threshold (default 0.01).

    Returns:
        Verdict: PASS if p >= alpha, FAIL otherwise.
    """
    return Verdict.PASS if p_value >= alpha else Verdict.FAIL


def count_patterns(data: NDArray[np.uint8], pattern_len: int) -> NDArray[np.int64]:
    """Count overlapping m-bit patterns via chunked sliding window.

    Args:
        data: 1-D array of 0s and 1s.
        pattern_len: Number of bits per pattern (m).

    Returns:
        NDArray[np.int64]: Frequency array of length 2^m, indexed by pattern value.
    """
    n_patterns = 2 ** pattern_len
    n = len(data)
    if n < pattern_len:
        return np.zeros(n_patterns, dtype=np.int64)
    powers = (2 ** np.arange(pattern_len - 1, -1, -1)).astype(np.int64)
    counts = np.zeros(n_patterns, dtype=np.int64)
    # chunk to bound peak memory at chunk * pattern_len * 8 bytes
    chunk = 500_000
    total_windows = n - pattern_len + 1
    for start in range(0, total_windows, chunk):
        end = min(start + chunk + pattern_len - 1, n)
        chunk_data = data[start:end]
        windows = np.lib.stride_tricks.sliding_window_view(chunk_data, pattern_len)
        values = (windows.astype(np.int64) * powers).sum(axis=1)
        counts += np.bincount(values, minlength=n_patterns).astype(np.int64)
    return counts


def running_sum(data: NDArray[np.uint8]) -> NDArray[np.int64]:
    """Convert 0/1 bits to -1/+1 and compute the cumulative sum.

    Args:
        data: 1-D array of 0s and 1s.

    Returns:
        NDArray[np.int64]: Cumulative sum of the mapped sequence.
    """
    mapped = 2 * data.astype(np.int64) - 1
    return np.cumsum(mapped)


def berlekamp_massey(bits: NDArray[np.uint8]) -> int:
    """Compute the linear complexity of a binary sequence via Berlekamp-Massey.

    Args:
        bits: 1-D array of 0s and 1s.

    Returns:
        int: Length of the shortest LFSR that generates the sequence.
    """
    n = len(bits)
    c = np.zeros(n, dtype=np.int64)
    b = np.zeros(n, dtype=np.int64)
    c[0] = 1
    b[0] = 1
    lc = 0
    m = -1

    for i in range(n):
        d = bits[i]
        for j in range(1, lc + 1):
            d ^= c[j] & bits[i - j]
        d &= 1
        if d == 1:
            t = c.copy()
            shift = i - m
            for j in range(n - shift):
                c[j + shift] ^= b[j]
            if 2 * lc <= i:
                lc = i + 1 - lc
                m = i
                b = t
    return lc
