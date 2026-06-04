"""Information-theoretic measures of randomness quality."""

from __future__ import annotations

import math
import zlib

import numpy as np
from numpy.typing import NDArray

from .base import StatisticalTest, TestResult, Verdict
from ._utils import bits_to_blocks, bits_to_bytes, p_from_chi2, p_from_z, verdict


def _shannon(counts: NDArray[np.int64]) -> float:
    """Shannon entropy in bits from a count array.

    Args:
        counts: Array of non-negative integer frequencies.

    Returns:
        float: Entropy in bits, 0.0 if all counts are zero.
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log2(p)))


# ── Shannon Entropy ───────────────────────────────────────────

class ShannonEntropyTest(StatisticalTest):
    """Shannon entropy of the empirical distribution over m-bit blocks.

    H = -Σ p_i log2(p_i). For perfectly random m-bit blocks, H = m.
    """

    def __init__(self, *, block_size: int = 8) -> None:
        """Configure the block size m for entropy measurement.

        Args:
            block_size: Number of bits per block (m).
        """
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'Shannon Entropy (m=8)'.
        """
        return f"Shannon Entropy (m={self._block_size})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Compute Shannon entropy over m-bit blocks and chi-squared test against uniform.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Entropy as statistic, chi-squared p-value, and verdict.
        """
        blocks = bits_to_blocks(data, self._block_size)
        if len(blocks) == 0:
            return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)
        k = 2 ** self._block_size
        n = len(blocks)
        counts = np.bincount(blocks, minlength=k)

        h = _shannon(counts)

        # chi-squared gof against uniform
        expected = n / k
        chi2 = float(np.sum((counts - expected) ** 2 / expected))
        df = k - 1
        pval = p_from_chi2(chi2, df)

        return TestResult(
            test_name=self.name,
            statistic=h,
            p_value=pval,
            verdict=verdict(pval),
        )


# ── Min-Entropy ───────────────────────────────────────────────

class MinEntropyTest(StatisticalTest):
    """Min-entropy: H_inf = -log2(max p_i).

    The most conservative entropy estimate — determined entirely by
    the most probable outcome. Used by NIST SP 800-90B for
    non-IID sources (e.g. raw TRNG/QRNG output).
    """

    def __init__(self, *, block_size: int = 8) -> None:
        """Configure the block size m for entropy measurement.

        Args:
            block_size: Number of bits per block (m).
        """
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'Min-Entropy (m=8)'.
        """
        return f"Min-Entropy (m={self._block_size})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Compute min-entropy from the most probable m-bit block value.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Min-entropy as statistic, chi-squared p-value, and verdict.
        """
        blocks = bits_to_blocks(data, self._block_size)
        if len(blocks) == 0:
            return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)
        k = 2 ** self._block_size
        n = len(blocks)
        counts = np.bincount(blocks, minlength=k)

        p_max = float(counts.max()) / n
        h_inf = -math.log2(p_max) if p_max > 0 else float(self._block_size)

        # chi-squared gof against uniform (same approach as Shannon)
        expected = n / k
        chi2 = float(np.sum((counts - expected) ** 2 / expected))
        df = k - 1
        pval = p_from_chi2(chi2, df)

        return TestResult(
            test_name=self.name,
            statistic=h_inf,
            p_value=pval,
            verdict=verdict(pval),
        )


# ── Rényi Entropy ─────────────────────────────────────────────

class RenyiEntropyTest(StatisticalTest):
    """Rényi entropy of order α: H_α = (1/(1-α)) log2(Σ p_i^α).

    Interpolates between Hartley (α→0), Shannon (α→1), collision
    (α=2), and min-entropy (α→∞).
    """

    def __init__(self, *, alpha: float = 2.0, block_size: int = 8) -> None:
        """Configure Renyi order alpha and block size m.

        Args:
            alpha: Renyi order (must not be 1.0 -- that's Shannon entropy).
            block_size: Number of bits per block (m).

        Raises:
            ValueError: If alpha is exactly 1.0.
        """
        if alpha == 1.0:
            raise ValueError("α=1 is Shannon entropy; use ShannonEntropyTest")
        self._alpha = alpha
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'Renyi Entropy (a=2.0, m=8)'.
        """
        return f"Rényi Entropy (α={self._alpha}, m={self._block_size})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Compute Renyi entropy of order alpha over m-bit blocks.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Renyi entropy as statistic, chi-squared p-value, and verdict.
        """
        blocks = bits_to_blocks(data, self._block_size)
        if len(blocks) == 0:
            return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)
        k = 2 ** self._block_size
        n = len(blocks)
        counts = np.bincount(blocks, minlength=k)

        p = counts / n
        p_nonzero = p[p > 0]
        h_alpha = (1.0 / (1.0 - self._alpha)) * math.log2(float(np.sum(p_nonzero ** self._alpha)))

        # chi-squared gof against uniform (same as Shannon)
        expected = n / k
        chi2 = float(np.sum((counts - expected) ** 2 / expected))
        df = k - 1
        pval = p_from_chi2(chi2, df)

        return TestResult(
            test_name=self.name,
            statistic=h_alpha,
            p_value=pval,
            verdict=verdict(pval),
        )


# ── Compression Ratio ─────────────────────────────────────────

class CompressionRatioTest(StatisticalTest):
    """Compression ratio via zlib/gzip.

    Truly random data is incompressible; a ratio significantly
    below 1.0 indicates structure. Simple but effective.
    """

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Compression Ratio'.
        """
        return "Compression Ratio"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Compress the data with zlib and check if the ratio is close to 1.0.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Compression ratio as statistic, z-based p-value, and verdict.
        """
        raw = bits_to_bytes(data).tobytes()
        compressed = zlib.compress(raw, level=9)
        ratio = len(compressed) / len(raw) if len(raw) > 0 else 1.0

        # normal approx: for random data ratio ≈ 1.0 (often slightly above due to headers)
        # empirical std ~0.02 for typical sizes, scale with sqrt(n)
        n_bytes = len(raw)
        # rough std estimate — shrinks with more data
        std_est = max(0.5 / math.sqrt(n_bytes), 0.001)
        z = (1.0 - ratio) / std_est  # lower ratio = more compressible = less random
        pval = p_from_z(z)

        return TestResult(
            test_name=self.name,
            statistic=ratio,
            p_value=pval,
            verdict=verdict(pval),
        )


# ── Kolmogorov Complexity Estimate ────────────────────────────

class LempelZivComplexityTest(StatisticalTest):
    """Lempel-Ziv complexity (LZ76 decomposition).

    Counts the number of distinct substrings when parsing left to
    right. For n random bits, complexity ~ n / log2(n).
    Related to Kolmogorov complexity (upper bound).
    """

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Lempel-Ziv Complexity'.
        """
        return "Lempel-Ziv Complexity"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Count distinct substrings via LZ76 decomposition and z-test the normalised complexity.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Normalised complexity as statistic, z-based p-value, and verdict.
        """
        n = len(data)
        if n == 0:
            return TestResult(self.name, 0.0, 0.5, verdict(0.5))

        # LZ76 decomposition — cap input at 100k to keep runtime manageable
        # (O(n²) algorithm, ~10s at 100k, impractical above ~500k)
        cap = min(n, 100_000)
        capped = data[:cap]
        cn = cap
        complexity = 1
        i = 0
        l = 1
        s = capped.tobytes()
        while i + l <= cn:
            # check if s[i:i+l] appears in s[0:i+l-1]
            if s.find(s[i:i + l], 0, i + l - 1) >= 0:
                l += 1
                if i + l > cn:
                    complexity += 1
            else:
                complexity += 1
                i += l
                l = 1

        # normalized complexity: c * log2(n) / n -> 1.0 asymptotically
        log2n = math.log2(cn) if cn > 1 else 1.0
        normalized = complexity * log2n / cn

        # the asymptotic mean has a positive bias that decays slowly;
        # use a conservative std that absorbs finite-size corrections
        expected_norm = 1.0
        std = max(0.7 / log2n, 0.002)
        z = (normalized - expected_norm) / std if std > 0 else 0.0
        pval = p_from_z(z)

        return TestResult(
            test_name=self.name,
            statistic=normalized,
            p_value=pval,
            verdict=verdict(pval),
        )


# ── Conditional Entropy ───────────────────────────────────────

class ConditionalEntropyTest(StatisticalTest):
    """Conditional entropy H(X_n | X_{n-1}, ..., X_{n-k}).

    Measures how much uncertainty remains about the next bit given
    the previous k bits. For IID random bits this equals 1.0.
    """

    def __init__(self, *, order: int = 3) -> None:
        """Configure the conditioning order k.

        Args:
            order: Number of preceding bits to condition on (k).
        """
        self._order = order

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'Conditional Entropy (k=3)'.
        """
        return f"Conditional Entropy (k={self._order})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Compute H(X_n | X_{n-1},...,X_{n-k}) and z-test deviation from 1.0.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Conditional entropy as statistic, z-based p-value, and verdict.
        """
        k = self._order
        n = len(data)
        if n <= k:
            return TestResult(self.name, 0.0, 0.5, verdict(0.5))

        # count (k+1)-grams and k-grams using chunked sliding windows
        kp1 = k + 1
        from ._utils import count_patterns
        counts_kp1 = count_patterns(data, kp1)
        counts_k = count_patterns(data, k)

        # H(X_n | context) = H(k+1-gram) - H(k-gram)
        h_kp1 = _shannon(counts_kp1)
        h_k = _shannon(counts_k)
        h_cond = h_kp1 - h_k

        # for IID bits, h_cond -> 1.0
        # approximate p-value from deviation
        num_kp1_grams = n - kp1 + 1
        std_est = 1.0 / math.sqrt(num_kp1_grams) if num_kp1_grams > 0 else 1.0
        z = (1.0 - h_cond) / std_est
        pval = p_from_z(z)

        return TestResult(
            test_name=self.name,
            statistic=h_cond,
            p_value=pval,
            verdict=verdict(pval),
        )


# ── Mutual Information ────────────────────────────────────────

class MutualInformationTest(StatisticalTest):
    """Mutual information I(X; Y) between non-overlapping halves.

    Split the sequence into two halves and measure shared information.
    Should be approximately zero for independent random bits.
    """

    def __init__(self, *, block_size: int = 4) -> None:
        """Configure the block size m for computing mutual information.

        The chi-squared approximation requires n >> k^2 where k = 2^m.
        With m=4, k^2 = 256 which is comfortably below typical sample
        counts. Larger block sizes need proportionally more data.

        Args:
            block_size: Number of bits per block (m).
        """
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'Mutual Information (m=4)'.
        """
        return f"Mutual Information (m={self._block_size})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Split data in half and test whether mutual information between halves is near zero.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: MI in bits as statistic, chi-squared p-value, and verdict.
        """
        m = self._block_size
        half = len(data) // 2
        # make both halves divisible by block_size
        half = half - (half % m)

        x_blocks = bits_to_blocks(data[:half], m)
        y_blocks = bits_to_blocks(data[half:half * 2], m)
        k = 2 ** m
        n = min(len(x_blocks), len(y_blocks))
        x_blocks = x_blocks[:n]
        y_blocks = y_blocks[:n]

        # marginals
        cx = np.bincount(x_blocks, minlength=k).astype(np.float64)
        cy = np.bincount(y_blocks, minlength=k).astype(np.float64)
        # joint
        joint_idx = x_blocks * k + y_blocks
        cxy = np.bincount(joint_idx, minlength=k * k).astype(np.float64).reshape(k, k)

        px = cx / n
        py = cy / n
        pxy = cxy / n

        # I(X;Y) = sum pxy * log2(pxy / (px * py)) where all > 0
        mi = 0.0
        outer = np.outer(px, py)
        mask = (pxy > 0) & (outer > 0)
        if mask.any():
            mi = float(np.sum(pxy[mask] * np.log2(pxy[mask] / outer[mask])))

        # 2*n*I ~ chi2 with (k-1)^2 df for large samples
        df = (k - 1) ** 2
        if df > 0 and n > 0:
            chi2_stat = 2.0 * n * mi * math.log(2)  # convert nats for chi2
            # actually 2*n*I (in nats) is chi2, so convert from bits
            pval = p_from_chi2(chi2_stat, df)
        else:
            pval = 1.0

        return TestResult(
            test_name=self.name,
            statistic=mi,
            p_value=pval,
            verdict=verdict(pval),
        )


# ── Topological Permutation Entropy ───────────────────────────

class PermutationEntropyTest(StatisticalTest):
    """Permutation entropy (Bandt & Pompe, 2002).

    Maps the sequence to ordinal patterns of length d and computes
    Shannon entropy over pattern frequencies. Robust to noise.
    """

    def __init__(self, *, order: int = 5, delay: int = 1) -> None:
        """Configure ordinal pattern length d and embedding delay tau.

        Args:
            order: Ordinal pattern length (d). Produces d! possible patterns.
            delay: Embedding delay between successive elements (tau).
        """
        self._order = order
        self._delay = delay

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'Permutation Entropy (d=5, tau=1)'.
        """
        return f"Permutation Entropy (d={self._order}, τ={self._delay})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Compute Shannon entropy over ordinal patterns and chi-squared test against uniform.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Permutation entropy as statistic, chi-squared p-value, and verdict.
        """
        d = self._order
        tau = self._delay

        # convert bits to byte-valued blocks so we have enough distinct values
        # for meaningful ordinal patterns (raw bits only give 0/1)
        values = bits_to_blocks(data, 8).astype(np.float64)
        # break ties with small seeded jitter (< 1, so it never reorders
        # distinct byte values); a monotone offset would bias the ordinal ranks
        rng = np.random.default_rng(0)
        values = values + rng.random(len(values)) * 0.5

        n = len(values)
        window_len = (d - 1) * tau + 1

        if n < window_len:
            return TestResult(self.name, 0.0, 0.5, verdict(0.5))

        # non-overlapping windows: the chi-squared over ordinal patterns assumes
        # independent observations, so windows must not share samples
        indices = np.arange(d) * tau
        n_windows = n // window_len
        starts = np.arange(n_windows) * window_len
        windows = values[starts[:, None] + indices[None, :]]

        # ordinal pattern: argsort each row, map to Lehmer code -> [0, d!)
        ranks = np.argsort(np.argsort(windows, axis=1), axis=1)
        n_possible = math.factorial(d)

        # Lehmer encoding: for each position, count how many later elements are smaller
        # this maps each permutation to a unique int in [0, d!)
        factorials = np.array([math.factorial(d - 1 - i) for i in range(d)])
        lehmer = np.zeros(n_windows, dtype=np.int64)
        for i in range(d):
            # how many elements after position i have smaller rank
            later = ranks[:, i + 1:]
            count_smaller = np.sum(later < ranks[:, i:i + 1], axis=1)
            lehmer += count_smaller * factorials[i]

        counts = np.bincount(lehmer, minlength=n_possible)[:n_possible]

        h_perm = _shannon(counts)
        h_max = math.log2(n_possible) if n_possible > 1 else 1.0

        # chi-squared against uniform over d! patterns
        n_obs = counts.sum()
        expected = n_obs / n_possible
        chi2 = float(np.sum((counts - expected) ** 2 / expected))
        df = n_possible - 1
        pval = p_from_chi2(chi2, df)

        return TestResult(
            test_name=self.name,
            statistic=h_perm,
            p_value=pval,
            verdict=verdict(pval),
        )


# ── Convenience ───────────────────────────────────────────────

def entropy_battery() -> list[StatisticalTest]:
    """Return all information-theoretic tests with default parameters.

    Returns:
        list[StatisticalTest]: 8 entropy/complexity test instances.
    """
    return [
        ShannonEntropyTest(),
        MinEntropyTest(),
        RenyiEntropyTest(alpha=2.0),
        CompressionRatioTest(),
        LempelZivComplexityTest(),
        ConditionalEntropyTest(),
        MutualInformationTest(),
        PermutationEntropyTest(),
    ]
