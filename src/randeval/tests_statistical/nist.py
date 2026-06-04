"""All 15 NIST SP 800-22 statistical tests for randomness.

Reference: NIST Special Publication 800-22 Revision 1a (2010),
"A Statistical Test Suite for Random and Pseudorandom Number
Generators for Cryptographic Applications".
"""

from __future__ import annotations

from math import erfc, sqrt, log2, floor

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaincc

from .base import StatisticalTest, TestResult, Verdict
from ._utils import proportion_ones, p_from_chi2, verdict, running_sum, berlekamp_massey


# ── Test 1: Frequency (Monobit) ──────────────────────────────

class FrequencyTest(StatisticalTest):
    """Test 1 — Frequency (Monobit).

    Tests whether the number of 0s and 1s are approximately equal,
    as expected for a truly random sequence.
    """

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'NIST 1: Frequency (Monobit)'.
        """
        return "NIST 1: Frequency (Monobit)"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Check whether the proportion of 0s and 1s is approximately equal.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Normalised deviation as statistic, p-value via erfc, and verdict.
        """
        n = len(data)
        s_n = np.sum(2 * data.astype(np.int64) - 1)
        stat = abs(s_n) / sqrt(n)
        p = erfc(stat / sqrt(2))
        return TestResult(self.name, float(stat), float(p), verdict(p))


# ── Test 2: Block Frequency ───────────────────────────────────

class BlockFrequencyTest(StatisticalTest):
    """Test 2 — Frequency within a Block.

    Divides the sequence into M-bit blocks and tests whether the
    proportion of 1s in each block is approximately M/2.
    """

    def __init__(self, *, block_size: int = 128) -> None:
        """Configure the block frequency test.

        Args:
            block_size: Number of bits per block (M in the NIST spec).
        """
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'NIST 2: Block Frequency (M=128)'.
        """
        return f"NIST 2: Block Frequency (M={self._block_size})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Test whether each M-bit block has roughly 50% ones.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        M = self._block_size
        n = len(data)
        N = n // M
        blocks = data[:N * M].reshape(N, M)
        proportions = blocks.mean(axis=1)
        chi2 = 4.0 * M * np.sum((proportions - 0.5) ** 2)
        p = p_from_chi2(chi2, N)
        return TestResult(self.name, float(chi2), float(p), verdict(p))


# ── Test 3: Runs ──────────────────────────────────────────────

class RunsTest(StatisticalTest):
    """Test 3 — Runs.

    Tests whether the number of uninterrupted runs of identical bits
    is consistent with a random sequence. A run is a maximal sequence
    of consecutive identical bits.
    """

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'NIST 3: Runs'.
        """
        return "NIST 3: Runs"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Test whether the number of runs of consecutive identical bits is as expected.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Normalised deviation as statistic, p-value via erfc, and verdict.
        """
        n = len(data)
        pi = proportion_ones(data)
        # pre-test
        if abs(pi - 0.5) >= 2.0 / sqrt(n):
            return TestResult(self.name, 0.0, 0.0, verdict(0.0))
        v_n = 1 + np.sum(data[:-1] != data[1:])
        num = abs(v_n - 2.0 * n * pi * (1 - pi))
        den = 2.0 * sqrt(2.0 * n) * pi * (1 - pi)
        stat = num / den
        # NIST 3.3: erfc is applied to the standardised statistic directly;
        # the sqrt(2) already lives in the denominator, no second one here.
        p = erfc(stat)
        return TestResult(self.name, float(stat), float(p), verdict(p))


# ── Test 4: Longest Run of Ones ───────────────────────────────

def _longest_run_params(n: int) -> tuple[int, int, int, list[float]]:
    """Pick (M, K, v_min, pi) for the longest-run test by sequence length.

    NIST 2.4.2 table: M=8 for 128<=n<6272, M=128 for 6272<=n<750000,
    M=10^4 for n>=750000.
    """
    if n < 6272:
        return 8, 3, 1, [0.2148, 0.3672, 0.2305, 0.1875]
    if n < 750000:
        return 128, 5, 4, [0.1174, 0.2430, 0.2493, 0.1752, 0.1027, 0.1124]
    return 10000, 6, 10, [0.0882, 0.2092, 0.2483, 0.1933, 0.1208, 0.0675, 0.0727]


class LongestRunOfOnesTest(StatisticalTest):
    """Test 4 — Longest Run of Ones in a Block.

    Tests whether the longest run of 1s within M-bit blocks is
    consistent with what's expected from a random sequence.
    """

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'NIST 4: Longest Run of Ones'.
        """
        return "NIST 4: Longest Run of Ones"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Check whether the longest run of 1s in each block matches the expected distribution.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        n = len(data)
        M, K, v_min, pi = _longest_run_params(n)

        N = n // M
        blocks = data[:N * M].reshape(N, M)

        # longest run of 1s in each block
        longest = np.zeros(N, dtype=np.int64)
        for i in range(N):
            run = 0
            best = 0
            for bit in blocks[i]:
                if bit == 1:
                    run += 1
                    if run > best:
                        best = run
                else:
                    run = 0
            longest[i] = best

        # bin the longest runs
        n_classes = len(pi)
        freq = np.zeros(n_classes, dtype=np.float64)
        for v in longest:
            idx = int(v) - v_min
            if idx < 0:
                idx = 0
            elif idx >= n_classes:
                idx = n_classes - 1
            freq[idx] += 1

        chi2 = np.sum((freq - N * np.array(pi)) ** 2 / (N * np.array(pi)))
        p = p_from_chi2(float(chi2), n_classes - 1)
        return TestResult(self.name, float(chi2), float(p), verdict(p))


# ── Test 5: Binary Matrix Rank ────────────────────────────────

class BinaryMatrixRankTest(StatisticalTest):
    """Test 5 — Binary Matrix Rank.

    Constructs matrices from sequential bits and checks whether
    the rank distribution matches that of random binary matrices.
    Detects linear dependence among fixed-length substrings.
    """

    def __init__(self, *, rows: int = 32, cols: int = 32) -> None:
        """Configure matrix dimensions for the rank test.

        Args:
            rows: Number of rows per binary matrix.
            cols: Number of columns per binary matrix.
        """
        self._rows = rows
        self._cols = cols

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'NIST 5: Binary Matrix Rank (32x32)'.
        """
        return f"NIST 5: Binary Matrix Rank ({self._rows}x{self._cols})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Build binary matrices from bits and test whether their GF(2) rank distribution is normal.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        M = self._rows
        Q = self._cols
        n = len(data)
        N = n // (M * Q)
        if N < 1:
            return TestResult(self.name, 0.0, 0.0, verdict(0.0))

        bits = data[:N * M * Q].reshape(N, M, Q)

        def _gf2_rank(mat: NDArray[np.uint8]) -> int:
            """Row-reduce over GF(2) and return the rank.

            Args:
                mat: 2-D binary matrix (will be copied before modification).

            Returns:
                int: Rank of the matrix over GF(2).
            """
            a = mat.copy()
            rows, cols = a.shape
            rank = 0
            for col in range(cols):
                pivot = None
                for row in range(rank, rows):
                    if a[row, col] == 1:
                        pivot = row
                        break
                if pivot is None:
                    continue
                a[[rank, pivot]] = a[[pivot, rank]]
                for row in range(rows):
                    if row != rank and a[row, col] == 1:
                        a[row] ^= a[rank]
                rank += 1
            return rank

        full = min(M, Q)
        ranks = np.array([_gf2_rank(bits[i]) for i in range(N)])

        f_full = np.sum(ranks == full)
        f_minus1 = np.sum(ranks == full - 1)
        f_rest = N - f_full - f_minus1

        # expected probabilities
        p_full = 0.2888
        p_minus1 = 0.5776
        p_rest = 0.1336

        chi2 = ((f_full - N * p_full) ** 2 / (N * p_full) +
                (f_minus1 - N * p_minus1) ** 2 / (N * p_minus1) +
                (f_rest - N * p_rest) ** 2 / (N * p_rest))
        p = p_from_chi2(float(chi2), 2)
        return TestResult(self.name, float(chi2), float(p), verdict(p))


# ── Test 6: Discrete Fourier Transform (Spectral) ────────────

class SpectralTest(StatisticalTest):
    """Test 6 — Discrete Fourier Transform (Spectral).

    Detects periodic features in the sequence by examining the peak
    heights in the DFT. Sensitive to repetitive patterns.
    """

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'NIST 6: DFT (Spectral)'.
        """
        return "NIST 6: DFT (Spectral)"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Apply DFT and check whether the number of peaks below the threshold is as expected.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Normalised d-statistic, p-value via erfc, and verdict.
        """
        n = len(data)
        x = 2.0 * data.astype(np.float64) - 1.0
        # rfft returns the first n//2+1 complex bins; the full fft is
        # symmetric so throwing away the negative half via fft is wasted
        # memory (>600MB for n=1e7).
        S = np.fft.rfft(x)
        half = n // 2
        magnitudes = np.abs(S[:half])
        del S, x
        T = sqrt(3.0 * n)
        n_obs = float(np.sum(magnitudes < T))
        n_expected = 0.95 * half
        d = (n_obs - n_expected) / sqrt(n * 0.95 * 0.05 / 4.0)
        p = erfc(abs(d) / sqrt(2))
        return TestResult(self.name, float(d), float(p), verdict(p))


# ── Test 7: Non-overlapping Template Matching ─────────────────

class NonOverlappingTemplateTest(StatisticalTest):
    """Test 7 — Non-overlapping Template Matching.

    Counts occurrences of a given m-bit template in non-overlapping
    blocks and tests against the expected count.
    """

    def __init__(self, *, template: list[int] | None = None, block_size: int | None = None) -> None:
        """Configure the bit template and optional block size.

        Args:
            template: Bit pattern to search for (default [0,0,0,0,0,0,0,0,1]).
            block_size: Block size M for partitioning. Auto-selected from data if None.
        """
        self._template = template or [0, 0, 0, 0, 0, 0, 0, 0, 1]
        self._block_size = block_size  # None = auto-select from data

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'NIST 7: Non-overlapping Template'.
        """
        return "NIST 7: Non-overlapping Template"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Count non-overlapping occurrences of a template in each block and chi-squared test.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        n = len(data)
        template = np.array(self._template, dtype=np.uint8)
        m = len(template)
        # auto-select block size: aim for ~8 blocks, minimum 2
        M = self._block_size if self._block_size is not None else max(m * 2, n // 8)
        if M < m:
            return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)
        N = n // M
        if N < 2:
            return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

        # count non-overlapping matches in each block
        counts = np.zeros(N, dtype=np.float64)
        for i in range(N):
            block = data[i * M:(i + 1) * M]
            j = 0
            while j <= M - m:
                if np.array_equal(block[j:j + m], template):
                    counts[i] += 1
                    j += m  # non-overlapping: skip past match
                else:
                    j += 1

        mu = (M - m + 1) / (2 ** m)
        sigma2 = M * (1.0 / (2 ** m) - (2 * m - 1) / (2 ** (2 * m)))
        chi2 = float(np.sum((counts - mu) ** 2 / sigma2))
        p = p_from_chi2(chi2, N)
        return TestResult(self.name, chi2, float(p), verdict(p))


# ── Test 8: Overlapping Template Matching ─────────────────────

class OverlappingTemplateTest(StatisticalTest):
    """Test 8 — Overlapping Template Matching.

    Similar to Test 7, but uses a sliding window (overlapping).
    Counts the number of times a template occurs in each block.
    """

    def __init__(self, *, template_length: int = 9, block_size: int = 1032) -> None:
        """Configure the all-ones template length and block size.

        Args:
            template_length: Length of the all-ones template (m).
            block_size: Block size M for partitioning.
        """
        self._template_length = template_length
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'NIST 8: Overlapping Template'.
        """
        return "NIST 8: Overlapping Template"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Count overlapping matches of an all-ones template and chi-squared test via Markov DP.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        from math import exp as mexp
        n = len(data)
        m = self._template_length
        M = self._block_size
        N = n // M
        K = 5  # bins: 0, 1, 2, 3, 4, >=5

        lam = float(M - m + 1) / (2 ** m)
        eta = lam / 2.0

        template = np.ones(m, dtype=np.uint8)

        # count overlapping matches in each block
        counts = np.zeros(N, dtype=np.int64)
        for i in range(N):
            block = data[i * M:(i + 1) * M]
            ct = 0
            for j in range(M - m + 1):
                if np.array_equal(block[j:j + m], template):
                    ct += 1
            counts[i] = ct

        freq = np.zeros(K + 1, dtype=np.float64)
        for cnt in counts:
            if cnt >= K:
                freq[K] += 1
            else:
                freq[cnt] += 1

        # compute exact probabilities via Markov chain DP
        # states: 0..m-1 = number of consecutive trailing 1s
        n_states = m
        dp = np.zeros((n_states, K + 1))
        dp[0][0] = 1.0

        for _ in range(M):
            new_dp = np.zeros((n_states, K + 1))
            for s in range(n_states):
                for c in range(K + 1):
                    if dp[s][c] == 0:
                        continue
                    p = dp[s][c] * 0.5
                    new_dp[0][c] += p  # saw 0, trailing 1s reset
                    if s < m - 1:
                        new_dp[s + 1][c] += p  # saw 1, no match yet
                    else:
                        # s == m-1: saw 1, completes a match; trailing
                        # m-1 bits are still all 1s (overlapping)
                        nc = min(c + 1, K)
                        new_dp[m - 1][nc] += p
            dp = new_dp

        pi = np.zeros(K + 1, dtype=np.float64)
        for c in range(K + 1):
            pi[c] = dp[:, c].sum()

        expected_counts = N * pi
        mask = expected_counts > 0
        chi2 = float(np.sum((freq[mask] - expected_counts[mask]) ** 2 / expected_counts[mask]))
        p = p_from_chi2(chi2, K)
        return TestResult(self.name, chi2, float(p), verdict(p))


# ── Test 9: Maurer's Universal Statistical ────────────────────

class MaurersUniversalTest(StatisticalTest):
    """Test 9 — Maurer's Universal Statistical Test.

    Measures compressibility. A non-random sequence will be more
    compressible, producing a lower test statistic.
    """

    def __init__(self, *, block_length: int = 7, init_blocks: int = 1280) -> None:
        """Configure L-bit block length and number of initialisation blocks Q.

        Args:
            block_length: Bits per block (L). Must be between 1 and 16 for the lookup table.
            init_blocks: Number of initialisation blocks (Q) to seed the table.
        """
        self._block_length = block_length
        self._init_blocks = init_blocks

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like "NIST 9: Maurer's Universal (L=7)".
        """
        return f"NIST 9: Maurer's Universal (L={self._block_length})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Measure compressibility via log-distances between repeated L-bit patterns.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Mean log-distance as statistic, p-value via erfc, and verdict.
        """
        L = self._block_length
        Q = self._init_blocks
        n = len(data)
        K = (n // L) - Q  # test blocks
        if K <= 0:
            return TestResult(self.name, 0.0, 0.0, verdict(0.0))

        # NIST table for expected value and variance
        ev_table = {
            1: (0.7326495, 0.690),
            2: (1.5374383, 1.338),
            3: (2.4016068, 1.901),
            4: (3.3112247, 2.358),
            5: (4.2534266, 2.705),
            6: (5.2177052, 2.954),
            7: (6.1962507, 3.125),
            8: (7.1836656, 3.238),
            9: (8.1764248, 3.311),
            10: (9.1723243, 3.356),
            11: (10.170032, 3.384),
            12: (11.168765, 3.401),
            13: (12.168070, 3.410),
            14: (13.167693, 3.416),
            15: (14.167488, 3.419),
            16: (15.167379, 3.421),
        }
        expected, variance = ev_table.get(L, (L * 0.7 + 0.8, L * 0.3 + 1.0))

        # init: last occurrence of each L-bit pattern
        last_occ = np.zeros(2 ** L, dtype=np.int64)
        for i in range(Q):
            val = 0
            for j in range(L):
                val = (val << 1) | int(data[i * L + j])
            last_occ[val] = i + 1

        total = 0.0
        for i in range(Q, Q + K):
            val = 0
            for j in range(L):
                val = (val << 1) | int(data[i * L + j])
            dist = i + 1 - last_occ[val]
            last_occ[val] = i + 1
            total += log2(dist)

        fn = total / K
        sigma = sqrt(variance / K)
        stat = abs(fn - expected) / sigma
        p = erfc(stat / sqrt(2))
        return TestResult(self.name, float(fn), float(p), verdict(p))


# ── Test 10: Linear Complexity ────────────────────────────────

class LinearComplexityTest(StatisticalTest):
    """Test 10 — Linear Complexity.

    Uses the Berlekamp-Massey algorithm to determine the length of
    the shortest LFSR that can generate each block. Random sequences
    should have high linear complexity.
    """

    def __init__(self, *, block_size: int = 500) -> None:
        """Configure the block size M for Berlekamp-Massey analysis.

        Args:
            block_size: Number of bits per block (M). Should be >= 500 for good results.
        """
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'NIST 10: Linear Complexity (M=500)'.
        """
        return f"NIST 10: Linear Complexity (M={self._block_size})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Compute LFSR linear complexity per block and chi-squared test the distribution.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        M = self._block_size
        n = len(data)
        N = n // M
        if N < 1:
            return TestResult(self.name, 0.0, 0.0, verdict(0.0))

        # Berlekamp-Massey is O(M^2) per block; NIST requires N>=200,
        # 1000 is a tight chi-squared without melting the CPU.
        N = min(N, 1000)
        blocks = data[:N * M].reshape(N, M)
        complexities = np.array([berlekamp_massey(blocks[i]) for i in range(N)])

        # T_i values per NIST spec section 2.10
        sign = (-1.0) ** M
        mu_exp = M / 2.0 + (9.0 + (-1) ** (M + 1)) / 36.0
        T = sign * (complexities - mu_exp) + 2.0 / 9.0

        # 7 bins: K=6 degrees of freedom
        # exact probabilities from NIST 2.10.5 (1/96, 1/32, 1/8, ... , 1/48)
        pi = np.array([0.010417, 0.031250, 0.125000, 0.500000, 0.250000, 0.062500, 0.020833])
        # bins: T<=-2.5, -2.5<T<=-1.5, -1.5<T<=-0.5, -0.5<T<=0.5, 0.5<T<=1.5, 1.5<T<=2.5, T>2.5
        freq = np.zeros(7, dtype=np.float64)
        for t in T:
            if t <= -2.5:
                freq[0] += 1
            elif t <= -1.5:
                freq[1] += 1
            elif t <= -0.5:
                freq[2] += 1
            elif t <= 0.5:
                freq[3] += 1
            elif t <= 1.5:
                freq[4] += 1
            elif t <= 2.5:
                freq[5] += 1
            else:
                freq[6] += 1

        chi2 = float(np.sum((freq - N * pi) ** 2 / (N * pi)))
        p = p_from_chi2(chi2, 6)
        return TestResult(self.name, chi2, float(p), verdict(p))


# ── Test 11: Serial ───────────────────────────────────────────

class SerialTest(StatisticalTest):
    """Test 11 — Serial.

    Tests whether every m-bit pattern appears with approximately
    equal frequency. Generalisation of the frequency test to
    overlapping m-bit patterns.
    """

    def __init__(self, *, pattern_length: int = 16) -> None:
        """Configure the overlapping pattern length m.

        Args:
            pattern_length: Number of bits per overlapping pattern (m).
        """
        self._pattern_length = pattern_length

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'NIST 11: Serial (m=16)'.
        """
        return f"NIST 11: Serial (m={self._pattern_length})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Test whether all m-bit overlapping patterns appear with equal frequency.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Delta-psi statistic, first p-value from gammaincc, and verdict.
        """
        m = self._pattern_length
        n = len(data)

        def _psi_sq(length: int) -> float:
            """Compute psi-squared statistic for overlapping patterns of given length.

            Args:
                length: Pattern length to evaluate.

            Returns:
                float: Psi-squared value, or 0.0 if length <= 0.
            """
            if length <= 0:
                return 0.0
            # extend data with wraparound, count in chunks to bound memory
            extended = np.concatenate([data, data[:length - 1]])
            powers = (2 ** np.arange(length - 1, -1, -1)).astype(np.int64)
            n_patterns = 2 ** length
            counts = np.zeros(n_patterns, dtype=np.int64)
            chunk = 500_000
            total_w = len(extended) - length + 1
            for start in range(0, total_w, chunk):
                end = min(start + chunk + length - 1, len(extended))
                windows = np.lib.stride_tricks.sliding_window_view(extended[start:end], length)
                vals = (windows.astype(np.int64) * powers).sum(axis=1)
                counts += np.bincount(vals, minlength=n_patterns).astype(np.int64)
            counts_f = counts.astype(np.float64)
            return float(np.sum(counts_f ** 2) * n_patterns / n - n)

        psi_m = _psi_sq(m)
        psi_m1 = _psi_sq(m - 1)
        psi_m2 = _psi_sq(m - 2)

        dpsi = psi_m - psi_m1
        d2psi = psi_m - 2 * psi_m1 + psi_m2

        p1 = float(gammaincc(2 ** (m - 2), dpsi / 2.0))
        p2 = float(gammaincc(2 ** (m - 3), d2psi / 2.0))

        # return first p-value as the primary result
        return TestResult(self.name, float(dpsi), float(p1), verdict(p1))


# ── Test 12: Approximate Entropy ──────────────────────────────

class ApproximateEntropyTest(StatisticalTest):
    """Test 12 — Approximate Entropy.

    Compares the frequency of overlapping blocks of two consecutive
    lengths (m and m+1). Similar to Serial but uses a different
    statistic.
    """

    def __init__(self, *, block_length: int = 10) -> None:
        """Configure the block length m for the approximate entropy comparison.

        Args:
            block_length: Overlapping pattern length (m).
        """
        self._block_length = block_length

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'NIST 12: Approximate Entropy (m=10)'.
        """
        return f"NIST 12: Approximate Entropy (m={self._block_length})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Compare overlapping pattern frequencies at lengths m and m+1.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared-like statistic, p-value, and verdict.
        """
        m = self._block_length
        n = len(data)

        def _phi(length: int) -> float:
            """Compute phi statistic (log-frequency sum) for overlapping patterns of given length.

            Args:
                length: Pattern length to evaluate.

            Returns:
                float: Phi value, or 0.0 if length <= 0.
            """
            if length <= 0:
                return 0.0
            extended = np.concatenate([data, data[:length - 1]])
            powers = (2 ** np.arange(length - 1, -1, -1)).astype(np.int64)
            n_patterns = 2 ** length
            counts_int = np.zeros(n_patterns, dtype=np.int64)
            chunk = 500_000
            total_w = len(extended) - length + 1
            for start in range(0, total_w, chunk):
                end = min(start + chunk + length - 1, len(extended))
                windows = np.lib.stride_tricks.sliding_window_view(extended[start:end], length)
                vals = (windows.astype(np.int64) * powers).sum(axis=1)
                counts_int += np.bincount(vals, minlength=n_patterns).astype(np.int64)
            counts = counts_int.astype(np.float64) / n
            # avoid log(0)
            mask = counts > 0
            return float(np.sum(counts[mask] * np.log(counts[mask])))

        phi_m = _phi(m)
        phi_m1 = _phi(m + 1)

        stat = 2.0 * n * (np.log(2) - (phi_m - phi_m1))
        p = p_from_chi2(float(stat), 2 ** m)
        return TestResult(self.name, float(stat), float(p), verdict(p))


# ── Test 13: Cumulative Sums ──────────────────────────────────

class CumulativeSumsTest(StatisticalTest):
    """Test 13 — Cumulative Sums (CUSUM).

    Converts bits to +1/-1 and computes the running sum. Tests
    whether the maximum excursion is too large. Can run forward
    or backward.
    """

    def __init__(self, *, forward: bool = True) -> None:
        """Configure scan direction.

        Args:
            forward: True for forward scan, False for backward.
        """
        self._forward = forward

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'NIST 13: Cumulative Sums (Forward)'.
        """
        direction = "Forward" if self._forward else "Backward"
        return f"NIST 13: Cumulative Sums ({direction})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Compute the maximum excursion of the cumulative +1/-1 sum.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Max |S| as statistic, p-value from NIST formula, and verdict.
        """
        from scipy.stats import norm
        n = len(data)
        seq = data if self._forward else data[::-1]
        S = running_sum(seq)
        z = float(np.max(np.abs(S)))

        # NIST section 2.13.4 formula
        sum1 = 0.0
        for k in range(int(floor((-n / z + 1) / 4.0)), int(floor((n / z - 1) / 4.0)) + 1):
            sum1 += norm.cdf((4 * k + 1) * z / sqrt(n)) - norm.cdf((4 * k - 1) * z / sqrt(n))
        sum2 = 0.0
        for k in range(int(floor((-n / z - 3) / 4.0)), int(floor((n / z - 1) / 4.0)) + 1):
            sum2 += norm.cdf((4 * k + 3) * z / sqrt(n)) - norm.cdf((4 * k + 1) * z / sqrt(n))
        p = 1.0 - sum1 + sum2

        return TestResult(self.name, float(z), float(p), verdict(p))


# ── Test 14: Random Excursions ────────────────────────────────

class RandomExcursionsTest(StatisticalTest):
    """Test 14 — Random Excursions.

    Counts the number of cycles having exactly k visits to a
    particular state in a cumulative sum random walk. Tests 8
    states: {-4, -3, -2, -1, +1, +2, +3, +4}.
    """

    def __init__(self, *, state: int = 1) -> None:
        """Configure the target state x for cycle-visit counting.

        Args:
            state: Target state in {-4,...,-1,+1,...,+4}.

        Raises:
            ValueError: If state is 0 or |state| > 4.
        """
        if state == 0 or abs(state) > 4:
            raise ValueError("State must be in {-4,-3,-2,-1,+1,+2,+3,+4}")
        self._state = state

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'NIST 14: Random Excursions (x=1)'.
        """
        return f"NIST 14: Random Excursions (x={self._state})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Count visits to state x per cycle of the cumulative sum random walk.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        n = len(data)
        S = running_sum(data)
        # prepend 0 and append 0 to form the walk
        walk = np.concatenate([[0], S, [0]])

        # find cycles (subsequences between zeros)
        zero_idx = np.where(walk == 0)[0]
        J = len(zero_idx) - 1
        if J < 50:
            return TestResult(self.name, 0.0, 0.0, verdict(0.0))

        x = self._state
        ax = abs(x)
        # pi_k for k visits to state x (NIST section 2.14, Table 9)
        # pi_0 = 1 - 1/(2|x|), pi_k = 1/(4x^2) * (1 - 1/(2|x|))^(k-1) for k>=1
        pi = np.zeros(6, dtype=np.float64)
        pi[0] = 1.0 - 1.0 / (2.0 * ax)
        for k in range(1, 5):
            pi[k] = 1.0 / (4.0 * ax * ax) * (1.0 - 1.0 / (2.0 * ax)) ** (k - 1)
        pi[5] = 1.0 - np.sum(pi[:5])

        # count visits to state x in each cycle
        freq = np.zeros(6, dtype=np.float64)  # bins: 0,1,2,3,4,>=5
        for c in range(J):
            cycle = walk[zero_idx[c]:zero_idx[c + 1] + 1]
            visits = int(np.sum(cycle == x))
            if visits >= 5:
                freq[5] += 1
            else:
                freq[visits] += 1

        expected_counts = J * pi
        mask = expected_counts > 0
        chi2 = float(np.sum((freq[mask] - expected_counts[mask]) ** 2 / expected_counts[mask]))
        p = p_from_chi2(chi2, 5)
        return TestResult(self.name, chi2, float(p), verdict(p))


# ── Test 15: Random Excursions Variant ────────────────────────

class RandomExcursionsVariantTest(StatisticalTest):
    """Test 15 — Random Excursions Variant.

    Counts the total number of times the cumulative sum random walk
    visits a particular state. Tests 18 states: {-9,...,-1,+1,...,+9}.
    """

    def __init__(self, *, state: int = 1) -> None:
        """Configure the target state x for visit counting.

        Args:
            state: Target state in {-9,...,-1,+1,...,+9}.

        Raises:
            ValueError: If state is 0 or |state| > 9.
        """
        if state == 0 or abs(state) > 9:
            raise ValueError("State must be in {-9,...,-1,+1,...,+9}")
        self._state = state

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'NIST 15: Random Excursions Variant (x=1)'.
        """
        return f"NIST 15: Random Excursions Variant (x={self._state})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Count total visits to state x in the random walk and compare to expected.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Normalised statistic, p-value via erfc, and verdict.
        """
        n = len(data)
        S = running_sum(data)
        walk = np.concatenate([[0], S, [0]])

        zero_idx = np.where(walk == 0)[0]
        J = len(zero_idx) - 1
        if J < 50:
            return TestResult(self.name, 0.0, 0.0, verdict(0.0))

        x = self._state
        # total visits to state x across all cycles
        visits = int(np.sum(walk[1:-1] == x))
        # NIST 2.15.4: denominator is sqrt(2*J*(4|x|-2)); erfc takes it directly.
        stat = abs(visits - J) / sqrt(2.0 * J * (4 * abs(x) - 2))
        p = erfc(stat)
        return TestResult(self.name, float(stat), float(p), verdict(p))


# ── Suite runner ──────────────────────────────────────────────

def nist_battery() -> list[StatisticalTest]:
    """Return all 15 NIST SP 800-22 tests with default parameters.

    Returns:
        list[StatisticalTest]: 16 test instances (CUSUM appears twice: forward + backward).
    """
    return [
        FrequencyTest(),
        BlockFrequencyTest(),
        RunsTest(),
        LongestRunOfOnesTest(),
        BinaryMatrixRankTest(),
        SpectralTest(),
        NonOverlappingTemplateTest(),
        OverlappingTemplateTest(),
        MaurersUniversalTest(),
        LinearComplexityTest(),
        SerialTest(),
        ApproximateEntropyTest(),
        CumulativeSumsTest(forward=True),
        CumulativeSumsTest(forward=False),
        RandomExcursionsTest(),
        RandomExcursionsVariantTest(),
    ]
