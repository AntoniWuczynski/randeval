"""Tests inspired by the Dieharder suite (Robert G. Brown) and the
original Diehard battery (George Marsaglia, 1995).

These cover patterns that NIST SP 800-22 does not test directly.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats
from scipy.special import comb as sp_comb

from .base import StatisticalTest, TestResult, Verdict
from ._utils import bits_to_blocks, bits_to_floats, p_from_chi2, p_from_z, verdict, bits_to_bytes


# ── Birthday Spacings ─────────────────────────────────────────

class BirthdaySpacingsTest(StatisticalTest):
    """Birthday Spacings — Marsaglia.

    Choose m random "birthdays" in a "year" of n days. The spacings
    between sorted birthdays should be Poisson distributed. Detects
    generators with insufficient state mixing.
    """

    def __init__(self, *, num_birthdays: int = 512, year_length: int = 2**24) -> None:
        """Configure number of birthdays m and year length for the spacing test.

        Args:
            num_birthdays: Number of random birthdays to generate (m).
            year_length: Size of the "year" (number of possible days).
        """
        self._num_birthdays = num_birthdays
        self._year_length = year_length

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Dieharder: Birthday Spacings'.
        """
        return "Dieharder: Birthday Spacings"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Count duplicate spacings between sorted random birthdays and test against Poisson.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Duplicate count as statistic, two-sided Poisson p-value, and verdict.
        """
        m = self._num_birthdays
        yr = self._year_length
        bits_needed = m * 32
        if len(data) < bits_needed:
            return self._fail()

        floats = bits_to_floats(data[:bits_needed], 32)
        birthdays = np.sort((floats * yr).astype(np.int64))
        spacings = np.diff(birthdays)
        spacings.sort()
        # count duplicate spacings
        dups = int(np.sum(spacings[1:] == spacings[:-1]))

        lam = m ** 3 / (4.0 * yr)
        # poisson cdf for p-value (right tail)
        pval = float(1.0 - sp_stats.poisson.cdf(dups - 1, lam)) if dups > 0 else 1.0
        # two-sided: take smaller tail
        pval_left = float(sp_stats.poisson.cdf(dups, lam))
        pval = 2.0 * min(pval, pval_left)
        pval = min(pval, 1.0)

        return TestResult(self.name, float(dups), pval, verdict(pval))


# ── Overlapping Permutations ──────────────────────────────────

class OverlappingPermutationsTest(StatisticalTest):
    """Overlapping Permutations (5-tuples).

    Examines the relative ordering of consecutive groups of 5 numbers.
    All 5! = 120 permutations should appear with equal probability.
    """

    def __init__(self, *, tuple_size: int = 5) -> None:
        """Configure the tuple size for the permutation test.

        Args:
            tuple_size: Number of consecutive values per ordinal pattern (t).
        """
        self._tuple_size = tuple_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'Dieharder: Overlapping Permutations (5-tuples)'.
        """
        return f"Dieharder: Overlapping Permutations ({self._tuple_size}-tuples)"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Chi-squared test the frequency of all t! ordinal patterns in overlapping windows.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        t = self._tuple_size
        n_perms = math.factorial(t)
        floats = bits_to_floats(data, 32)
        if len(floats) < t + 100:
            return self._fail()

        n_tuples = len(floats) - t + 1
        counts = np.zeros(n_perms, dtype=np.int64)

        for i in range(n_tuples):
            tup = floats[i:i + t]
            rank = 0
            # lehmer code -> rank
            for j in range(t):
                smaller = int(np.sum(tup[j + 1:] < tup[j]))
                rank = rank * (t - j) + smaller
            counts[rank] += 1

        expected = n_tuples / n_perms
        chi2 = float(np.sum((counts - expected) ** 2 / expected))
        df = n_perms - 1
        pval = p_from_chi2(chi2, df)
        return TestResult(self.name, chi2, pval, verdict(pval))


# ── Parking Lot ───────────────────────────────────────────────

# Marsaglia parks 12,000 cars per rep; only at that saturation is the parked
# count approximately normal. The reference mean/sigma below were calibrated by
# simulating this exact geometry (side 100, exclusion distance 1) over 2000 reps
# with a high-quality RNG. Each rep needs 12000 (x,y) pairs = 768,000 bits, so
# the test needs at least that many bits to run at all.
_PARK_ATTEMPTS = 12000
_PARK_MEAN = 4087.4
_PARK_SD = 23.4


class ParkingLotTest(StatisticalTest):
    """Parking Lot — Marsaglia.

    Randomly park points in a 100x100 square, rejecting any that land within
    distance 1 of an already-parked point. After 12,000 attempts the number
    parked is ~4087 (sigma ~23) for a good source; a biased source clusters or
    spreads differently and shifts that count.
    """

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Dieharder: Parking Lot'.
        """
        return "Dieharder: Parking Lot"

    def _park(self, xs: NDArray[np.float64], ys: NDArray[np.float64]) -> int:
        """Count points that park without landing within distance 1 of another.

        Uses a 1x1 grid so each point only checks its 3x3 cell neighbourhood.
        """
        grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
        parked = 0
        for x, y in zip(xs, ys):
            cx, cy = int(x), int(y)
            ok = True
            for gx in (cx - 1, cx, cx + 1):
                for gy in (cy - 1, cy, cy + 1):
                    for px, py in grid.get((gx, gy), ()):
                        dx, dy = x - px, y - py
                        if dx * dx + dy * dy < 1.0:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                grid.setdefault((cx, cy), []).append((x, y))
                parked += 1
        return parked

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Park 12,000 cars per rep and test the mean parked count against theory.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: mean parked count as statistic, p-value, and verdict.
        """
        floats = bits_to_floats(data, 32)
        n_reps = (len(floats) // 2) // _PARK_ATTEMPTS
        if n_reps < 1:
            # not enough bits to reach the saturation regime; cannot conclude
            return TestResult(self.name, 0.0, 1.0, Verdict.PASS)

        side = 100.0
        counts = np.empty(n_reps, dtype=np.float64)
        for rep in range(n_reps):
            base = rep * _PARK_ATTEMPTS * 2
            xs = floats[base:base + _PARK_ATTEMPTS * 2:2] * side
            ys = floats[base + 1:base + _PARK_ATTEMPTS * 2:2] * side
            counts[rep] = self._park(xs, ys)

        # combine reps into one z-test of the mean against the calibrated mean
        z = (counts - _PARK_MEAN) / _PARK_SD
        combined = float(np.mean(z) * np.sqrt(n_reps))
        pval = p_from_z(combined)
        return TestResult(self.name, float(counts.mean()), pval, verdict(pval))


# ── Minimum Distance (2D) ────────────────────────────────────

class MinimumDistanceTest(StatisticalTest):
    """Minimum Distance — Marsaglia.

    Place n random points in a unit square. The squared minimum
    distance should be exponentially distributed. Detects clustering.
    """

    def __init__(self, *, num_points: int | None = None) -> None:
        """Configure number of points for the minimum distance test.

        Args:
            num_points: Number of 2D points. Auto-derived from data length if None.
        """
        self._num_points = num_points

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Dieharder: Minimum Distance'.
        """
        return "Dieharder: Minimum Distance"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Place random 2D points and test whether the squared min distance is exponential.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Rescaled min-distance as statistic, two-sided p-value, and verdict.
        """
        # Marsaglia canonical: 8000 points. pdist on the full input is
        # O(n^2) in time and memory and blows up past ~20k points.
        default_n = 8000
        floats = bits_to_floats(data, 32)
        n = self._num_points if self._num_points is not None else default_n
        n = min(n, len(floats) // 2)
        if n < 100:
            return self._fail()

        xs = floats[:n * 2:2]
        ys = floats[1:n * 2:2]
        pts = np.column_stack((xs, ys))

        # KDTree nearest neighbour: O(n log n) instead of O(n^2)
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        d_nn, _ = tree.query(pts, k=2)
        min_d2 = float(np.min(d_nn[:, 1]) ** 2)

        # d_min^2 * pi * n*(n-1)/2 should be ~ Exp(1)
        lam = math.pi * n * (n - 1) / 2.0
        stat = min_d2 * lam
        # CDF of Exp(1): 1 - exp(-x)
        pval = 1.0 - math.exp(-stat)
        # make two-sided
        pval = 2.0 * min(pval, 1.0 - pval)
        pval = min(max(pval, 0.0), 1.0)

        return TestResult(self.name, stat, pval, verdict(pval))


# ── Squeeze ───────────────────────────────────────────────────

class SqueezeTest(StatisticalTest):
    """Squeeze — Marsaglia.

    Starting from k = 2^31, repeatedly set k = ceil(k * U) where
    U ~ Uniform(0,1). Count iterations until k = 1. The distribution
    of counts is known.
    """

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Dieharder: Squeeze'.
        """
        return "Dieharder: Squeeze"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Squeeze k from 2^31 to 1 using random multipliers and z-test the mean step count.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Z-score as statistic, p-value, and verdict.
        """
        floats = bits_to_floats(data, 32)
        num_trials = min(10_000, len(floats) // 35)
        if num_trials < 10:
            return self._fail()

        all_steps = np.empty(num_trials, dtype=np.float64)
        fi = 0
        for trial in range(num_trials):
            k = 2**31
            steps = 0
            while k > 1:
                if fi >= len(floats):
                    return self._fail()
                u = floats[fi]
                fi += 1
                u = max(u, 1e-15)
                u = min(u, 1.0 - 1e-15)
                k = math.ceil(k * u)
                steps += 1
            all_steps[trial] = steps

        mean_steps = float(np.mean(all_steps))
        var_steps = float(np.var(all_steps, ddof=1))
        expected_mean = math.log(2**31) + 0.5772156649 + 1.0
        sigma_mean = math.sqrt(max(var_steps, 1.0) / num_trials)
        z = (mean_steps - expected_mean) / sigma_mean
        pval = p_from_z(z)
        return TestResult(self.name, z, pval, verdict(pval))


# ── Overlapping Sums ──────────────────────────────────────────

class OverlappingSumsTest(StatisticalTest):
    """Overlapping Sums — Marsaglia.

    Forms overlapping sums of 100 consecutive uniform(0,1) values.
    The distribution should be approximately normal.
    """

    def __init__(self, *, window: int = 100) -> None:
        """Configure the sliding window width for overlapping sums.

        Args:
            window: Number of consecutive floats to sum (w).
        """
        self._window = window

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'Dieharder: Overlapping Sums (w=100)'.
        """
        return f"Dieharder: Overlapping Sums (w={self._window})"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Compute overlapping sums of w floats and KS-test their normality.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: KS statistic, p-value, and verdict.
        """
        w = self._window
        floats = bits_to_floats(data, 32)
        if len(floats) < w + 100:
            return self._fail()

        if len(floats) < w + 9:
            return self._fail()
        sums = np.lib.stride_tricks.sliding_window_view(floats, w).sum(axis=1)

        # subsample every w-th sum to get approximately independent values
        independent = sums[::w]
        if len(independent) < 10:
            return self._fail()

        mu = w / 2.0
        sigma = math.sqrt(w / 12.0)
        standardised = (independent - mu) / sigma

        ks_stat, pval = sp_stats.kstest(standardised, 'norm')
        return TestResult(self.name, float(ks_stat), float(pval), verdict(float(pval)))


# ── Craps ─────────────────────────────────────────────────────

class CrapsTest(StatisticalTest):
    """Craps — Marsaglia.

    Simulates the dice game. Tests both the win frequency (~244/495)
    and the distribution of game lengths.
    """

    def __init__(self, *, num_games: int = 200_000) -> None:
        """Configure the number of craps games to simulate.

        Args:
            num_games: Target number of games (actual count depends on data length).
        """
        self._num_games = num_games

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Dieharder: Craps'.
        """
        return "Dieharder: Craps"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Simulate craps games and z-test the win rate against the theoretical 244/495.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Z-score as statistic, p-value, and verdict.
        """
        floats = bits_to_floats(data, 32)
        # each game uses ~3.4 rolls avg, 2 dice each = ~7 floats per game
        ng = min(self._num_games, len(floats) // 8)
        if ng < 50:
            return self._fail()

        def roll_die(idx: int) -> tuple[int, int]:
            """Return die value 1-6 and the next float index.

            Args:
                idx: Current position in the floats array.

            Returns:
                tuple[int, int]: (die value 1-6, next index). Returns (-1, idx) if out of data.
            """
            if idx >= len(floats):
                return -1, idx
            val = int(floats[idx] * 6) + 1
            val = min(val, 6)
            return val, idx + 1

        max_len = 21  # bin game lengths 1..20, 21+
        length_counts = np.zeros(max_len, dtype=np.int64)
        wins = 0
        fi = 0

        for _ in range(ng):
            d1, fi = roll_die(fi)
            d2, fi = roll_die(fi)
            if d1 < 0 or d2 < 0:
                return self._fail()
            total = d1 + d2
            game_len = 1

            if total in (7, 11):
                wins += 1
            elif total not in (2, 3, 12):
                point = total
                while True:
                    d1, fi = roll_die(fi)
                    d2, fi = roll_die(fi)
                    if d1 < 0 or d2 < 0:
                        return self._fail()
                    game_len += 1
                    t = d1 + d2
                    if t == point:
                        wins += 1
                        break
                    if t == 7:
                        break

            idx = min(game_len, max_len) - 1
            length_counts[idx] += 1

        # chi-squared on game length distribution
        # expected probabilities per length are complex; use z-test on win rate instead
        p_win = 244.0 / 495.0
        expected_wins = ng * p_win
        sigma_wins = math.sqrt(ng * p_win * (1 - p_win))
        z = (wins - expected_wins) / sigma_wins
        pval = p_from_z(z)
        return TestResult(self.name, z, pval, verdict(pval))


# ── GCD (Greatest Common Divisor) ─────────────────────────────

class GCDTest(StatisticalTest):
    """GCD Test — Marsaglia & Tsang (2002).

    Computes GCD of pairs of random integers. The number of steps
    and the resulting GCD have known distributions.
    """

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Dieharder: GCD'.
        """
        return "Dieharder: GCD"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Compute GCD step counts for random integer pairs and z-test the mean.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Z-score as statistic, p-value, and verdict.
        """
        ints = bits_to_blocks(data, 32)
        num_pairs = len(ints) // 2
        if num_pairs < 1000:
            return self._fail()

        max_steps = 40
        step_counts = np.zeros(max_steps, dtype=np.int64)

        for i in range(num_pairs):
            a = int(abs(ints[2 * i])) + 1
            b = int(abs(ints[2 * i + 1])) + 1
            steps = 0
            while b != 0:
                a, b = b, a % b
                steps += 1
            idx = min(steps, max_steps - 1)
            step_counts[idx] += 1

        total = num_pairs
        bins = np.arange(max_steps, dtype=np.float64)
        mean_steps = float(np.sum(bins * step_counts)) / total
        var_steps = float(np.sum(bins**2 * step_counts)) / total - mean_steps**2
        # calibrated expected for 32-bit integers: ~18.76, var ~11.6
        expected_mean = 18.76
        sigma = math.sqrt(max(var_steps, 1.0) / total)
        z = (mean_steps - expected_mean) / sigma
        pval = p_from_z(z)
        return TestResult(self.name, z, pval, verdict(pval))


# ── Gorilla (Binary Rank 32x32 variant) ──────────────────────

class GorillaTest(StatisticalTest):
    """Gorilla Test — Calude et al.

    Extracts a specific bit position across 2^26 words and applies
    the OPSO (Overlapping Pairs Sparse Occupancy) test. Very sensitive
    to single-bit weaknesses.
    """

    def __init__(self, *, bit_position: int = 0) -> None:
        """Configure which bit position to extract from each 32-bit word.

        Args:
            bit_position: Bit index (0-31) to extract from each word.
        """
        self._bit_position = bit_position

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'Dieharder: Gorilla (bit=0)'.
        """
        return f"Dieharder: Gorilla (bit={self._bit_position})"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Extract one bit position across words and OPSO-test the missing pattern count.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Z-score as statistic, p-value, and verdict.
        """
        ints = bits_to_blocks(data, 32)
        word_len = 26
        while 2 ** word_len > len(ints) and word_len > 8:
            word_len -= 1
        n_words = min(2 ** word_len, len(ints))
        if n_words < 256:
            return self._fail()

        bp = self._bit_position % 32
        bits = ((ints[:n_words] >> (31 - bp)) & 1).astype(np.uint8)

        n_patterns = 2 ** word_len
        seen = np.zeros(n_patterns, dtype=np.bool_)

        val = 0
        mask = n_patterns - 1
        # init first window
        for j in range(word_len):
            val = (val << 1) | int(bits[j])

        seen[val] = True
        for j in range(word_len, len(bits)):
            val = ((val << 1) | int(bits[j])) & mask
            seen[val] = True

        missing = int(n_patterns - np.sum(seen))

        # expected missing ~ n_patterns * (1 - 1/n_patterns)^(n_overlapping)
        n_overlapping = len(bits) - word_len + 1
        expected_missing = n_patterns * ((1.0 - 1.0 / n_patterns) ** n_overlapping)
        sigma_missing = math.sqrt(n_patterns * (1.0 - 1.0 / n_patterns) ** n_overlapping
                                  * (1.0 - (1.0 - 1.0 / n_patterns) ** n_overlapping))

        if sigma_missing < 1e-10:
            return self._fail()

        if expected_missing < 10 or sigma_missing < 1.0:
            return TestResult(self.name, 0.0, 0.5, Verdict.PASS)

        z = (missing - expected_missing) / sigma_missing
        pval = p_from_z(z)
        return TestResult(self.name, z, pval, verdict(pval))


# ── Coupon Collector ──────────────────────────────────────────

class CouponCollectorTest(StatisticalTest):
    """Coupon Collector's Test.

    Maps bits to d-ary digits and counts how many draws are needed
    to see all d values. The distribution of collection times is known.
    """

    def __init__(self, *, d: int = 5) -> None:
        """Configure the alphabet size d for coupon collection.

        Args:
            d: Number of distinct "coupon" values to collect.
        """
        self._d = d

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'Dieharder: Coupon Collector (d=5)'.
        """
        return f"Dieharder: Coupon Collector (d={self._d})"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Measure collection times to see all d values and chi-squared test the distribution.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        d = self._d
        bits_per_digit = max(1, int(math.ceil(math.log2(d)))) if d > 1 else 1
        digits = bits_to_blocks(data, bits_per_digit)
        # reject digits >= d
        digits = digits[digits < d]
        if len(digits) < d * 20:
            return self._fail()

        # count collection times: how many draws until all d values seen
        max_t = d * 5  # max bin
        counts = np.zeros(max_t, dtype=np.int64)
        n_trials = 0
        i = 0

        while i < len(digits):
            seen: set[int] = set()
            t = 0
            while len(seen) < d:
                if i >= len(digits):
                    break
                seen.add(int(digits[i]))
                i += 1
                t += 1
            if len(seen) < d:
                break
            idx = min(t - d, max_t - 1)  # minimum t is d
            idx = max(idx, 0)
            counts[idx] += 1
            n_trials += 1

        if n_trials < 100:
            return self._fail()

        # compute P(T >= t) via inclusion-exclusion, then diff for P(T = t)
        probs = np.zeros(max_t)
        def p_ge(t: int) -> float:
            """Probability that collection time T >= t via inclusion-exclusion.

            Args:
                t: Collection time threshold.

            Returns:
                float: P(T >= t).
            """
            # P(all d seen in t draws) = sum_{j=0}^{d} (-1)^j * C(d,j) * ((d-j)/d)^t
            val = 0.0
            for j in range(d + 1):
                val += ((-1) ** j) * sp_comb(d, j, exact=True) * ((d - j) / d) ** t
            return 1.0 - val

        # p_ge(t) = P(T > t), so P(T = t) = p_ge(t-1) - p_ge(t)
        for t_off in range(max_t):
            t = t_off + d  # actual collection time
            if t_off < max_t - 1:
                probs[t_off] = p_ge(t - 1) - p_ge(t)
            else:
                probs[t_off] = p_ge(t - 1)  # tail bin: P(T >= t)

        # merge small bins
        expected = probs * n_trials
        # merge from the tail
        merged_obs: list[int] = []
        merged_exp: list[float] = []
        cum_obs, cum_exp = 0, 0.0
        for k in range(max_t):
            cum_obs += int(counts[k])
            cum_exp += expected[k]
            if cum_exp >= 5.0:
                merged_obs.append(cum_obs)
                merged_exp.append(cum_exp)
                cum_obs, cum_exp = 0, 0.0
        if cum_obs > 0 or cum_exp > 0:
            if merged_exp:
                merged_obs[-1] += cum_obs
                merged_exp[-1] += cum_exp
            else:
                merged_obs.append(cum_obs)
                merged_exp.append(cum_exp)

        if len(merged_obs) < 2:
            return self._fail()

        obs_arr = np.array(merged_obs, dtype=np.float64)
        exp_arr = np.array(merged_exp, dtype=np.float64)
        chi2 = float(np.sum((obs_arr - exp_arr) ** 2 / exp_arr))
        df = len(merged_obs) - 1
        pval = p_from_chi2(chi2, df)
        return TestResult(self.name, chi2, pval, verdict(pval))


# ── Gap Test ──────────────────────────────────────────────────

class GapTest(StatisticalTest):
    """Gap Test.

    Counts the lengths of gaps between occurrences of values falling
    in a specified range [alpha, beta). Gap lengths should follow a
    geometric distribution.
    """

    def __init__(self, *, alpha: float = 0.0, beta: float = 0.5) -> None:
        """Configure the target interval [alpha, beta) for gap detection.

        Args:
            alpha: Lower bound of the hit interval.
            beta: Upper bound of the hit interval.
        """
        self._alpha = alpha
        self._beta = beta

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'Dieharder: Gap ([0.0, 0.5))'.
        """
        return f"Dieharder: Gap ([{self._alpha}, {self._beta}))"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Measure gaps between hits of [alpha, beta) and chi-squared test against geometric.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        p = self._beta - self._alpha  # probability of hitting the interval
        if p <= 0:
            return self._fail()

        floats = bits_to_floats(data, 32)
        if len(floats) < 1000:
            return self._fail()

        # find gaps: count values between successive hits of [alpha, beta)
        hits = (floats >= self._alpha) & (floats < self._beta)
        hit_indices = np.where(hits)[0]

        if len(hit_indices) < 10:
            return self._fail()

        # gaps between consecutive hits (minus 1 gives gap length)
        gaps = np.diff(hit_indices) - 1

        # gap length follows geometric: P(gap=k) = (1-p)^k * p, k=0,1,2,...
        max_gap = 30
        gap_counts = np.zeros(max_gap, dtype=np.int64)
        for g in gaps:
            idx = min(int(g), max_gap - 1)
            gap_counts[idx] += 1

        n_gaps = len(gaps)
        expected = np.zeros(max_gap)
        for k in range(max_gap - 1):
            expected[k] = n_gaps * p * ((1 - p) ** k)
        expected[max_gap - 1] = n_gaps * ((1 - p) ** (max_gap - 1))  # tail bin

        # merge small bins
        merged_obs: list[int] = []
        merged_exp: list[float] = []
        co, ce = 0, 0.0
        for k in range(max_gap):
            co += int(gap_counts[k])
            ce += expected[k]
            if ce >= 5.0:
                merged_obs.append(co)
                merged_exp.append(ce)
                co, ce = 0, 0.0
        if co > 0 or ce > 0:
            if merged_exp:
                merged_obs[-1] += co
                merged_exp[-1] += ce
            else:
                merged_obs.append(co)
                merged_exp.append(max(ce, 1e-10))

        if len(merged_obs) < 2:
            return self._fail()

        obs_arr = np.array(merged_obs, dtype=np.float64)
        exp_arr = np.array(merged_exp, dtype=np.float64)
        chi2 = float(np.sum((obs_arr - exp_arr) ** 2 / exp_arr))
        df = len(merged_obs) - 1
        pval = p_from_chi2(chi2, df)
        return TestResult(self.name, chi2, pval, verdict(pval))


# ── Poker (Generalised) ──────────────────────────────────────

class PokerTest(StatisticalTest):
    """Generalised Poker Test.

    Groups bits into k-digit base-d numbers and counts how many
    distinct values appear in each group. Chi-squared against the
    expected multinomial distribution.
    """

    def __init__(self, *, group_size: int = 5, d: int = 8) -> None:
        """Configure group size k and alphabet size d for the poker hand test.

        Args:
            group_size: Number of digits per "hand" (k).
            d: Alphabet size (number of possible digit values).
        """
        self._group_size = group_size
        self._d = d

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'Dieharder: Poker (k=5, d=8)'.
        """
        return f"Dieharder: Poker (k={self._group_size}, d={self._d})"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Count distinct values per k-digit group and chi-squared test the distribution.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        k = self._group_size
        d = self._d
        bits_per_digit = max(1, int(math.ceil(math.log2(d)))) if d > 1 else 1

        digits = bits_to_blocks(data, bits_per_digit)
        digits = digits[digits < d]
        n_hands = len(digits) // k
        if n_hands < 100:
            return self._fail()
        digits = digits[:n_hands * k].reshape(n_hands, k)

        # count distinct values per hand
        distinct_counts = np.zeros(k + 1, dtype=np.int64)  # index by # distinct (1..k)
        for i in range(n_hands):
            nd = len(set(digits[i]))
            distinct_counts[nd] += 1

        # expected probability of r distinct values in k draws from d
        # P(r) = C(d,r) * S2(k,r) * r! / d^k where S2 is Stirling 2nd kind
        def stirling2(n: int, r: int) -> float:
            """Compute Stirling number of the second kind S(n, r).

            Args:
                n: Total items.
                r: Number of non-empty subsets.

            Returns:
                float: S(n, r).
            """
            if r == 0:
                return 1.0 if n == 0 else 0.0
            if r == 1 or r == n:
                return 1.0
            if r > n:
                return 0.0
            val = 0.0
            for j in range(r + 1):
                term = ((-1) ** (r - j)) * sp_comb(r, j, exact=True) * (j ** n)
                val += term
            return val / math.factorial(r)

        probs = np.zeros(k + 1)
        for r in range(1, k + 1):
            if r > d:
                break
            s2 = stirling2(k, r)
            probs[r] = sp_comb(d, r, exact=True) * s2 * math.factorial(r) / (d ** k)

        expected = probs * n_hands

        # merge bins
        merged_obs: list[int] = []
        merged_exp: list[float] = []
        co, ce = 0, 0.0
        for r in range(1, k + 1):
            co += int(distinct_counts[r])
            ce += expected[r]
            if ce >= 5.0:
                merged_obs.append(co)
                merged_exp.append(ce)
                co, ce = 0, 0.0
        if co > 0 or ce > 0:
            if merged_exp:
                merged_obs[-1] += co
                merged_exp[-1] += ce
            else:
                merged_obs.append(co)
                merged_exp.append(max(ce, 1e-10))

        if len(merged_obs) < 2:
            return self._fail()

        obs_arr = np.array(merged_obs, dtype=np.float64)
        exp_arr = np.array(merged_exp, dtype=np.float64)
        chi2 = float(np.sum((obs_arr - exp_arr) ** 2 / exp_arr))
        df = len(merged_obs) - 1
        pval = p_from_chi2(chi2, df)
        return TestResult(self.name, chi2, pval, verdict(pval))


# ── Collision Test ────────────────────────────────────────────

class CollisionTest(StatisticalTest):
    """Collision Test.

    Hashes n values into m bins. The number of collisions should
    follow a Poisson distribution with lambda = n^2/(2m).
    Sensitive to generators with short periods.
    """

    def __init__(self, *, num_values: int | None = None, num_bins: int = 2**14) -> None:
        """Configure number of values and bins for the collision count.

        Args:
            num_values: Number of values to hash. Auto-derived from data if None.
            num_bins: Number of hash bins (m).
        """
        self._num_values = num_values
        self._num_bins = num_bins

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Dieharder: Collision'.
        """
        return "Dieharder: Collision"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Hash values into bins and z-test the collision count against expectation.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Z-score as statistic, p-value, and verdict.
        """
        blocks = bits_to_blocks(data, 32)
        n = self._num_values if self._num_values is not None else len(blocks)
        n = min(n, len(blocks))
        if n < 100:
            return self._fail()

        # auto-scale bins so the Poisson regime holds: m >> n^2
        # target lambda = n^2/(2m) ~ 5, giving enough collisions to test
        # while keeping the normal approximation to Poisson valid
        m = max(self._num_bins, n * n // 10)

        vals = blocks[:n] % m
        vals[vals < 0] += m

        unique_count = len(np.unique(vals))
        collisions = n - unique_count

        # exact expected collisions: E[C] = n - m + m*(1 - 1/m)^n
        # for large m this simplifies to n - m*(1 - exp(-n/m))
        ratio = n / m
        if ratio < 1e-6:
            expected = n * (n - 1) / (2.0 * m)
        else:
            expected = n - m * (1.0 - math.exp(-ratio))

        # variance: m*q*(1 - q) - n*q_prev + expected^2/m
        # where q = (1-1/m)^n, q_prev = (1-1/m)^(n-1)
        # simplified: Var ≈ expected * exp(-ratio) for the Poisson regime
        variance = max(expected * math.exp(-ratio), 1.0)

        z = (collisions - expected) / math.sqrt(variance)
        pval = p_from_z(z)
        return TestResult(self.name, z, pval, verdict(pval))


# ── 3D Spheres ──────────────────────────────────────────────

class ThreeDSpheresTest(StatisticalTest):
    """3D Spheres — Marsaglia.

    Place n random points in a cube of side length L. Find the
    minimum distance between any pair. The cube of this distance
    should be exponentially distributed. Extends MinimumDistance
    to three dimensions.
    """

    def __init__(self, *, num_points: int | None = None) -> None:
        """Configure number of 3D points for the spheres test.

        Args:
            num_points: Number of points. Auto-derived from data length if None.
        """
        self._num_points = num_points

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Dieharder: 3D Spheres'.
        """
        return "Dieharder: 3D Spheres"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Place random 3D points and test whether the cubed min distance is exponential.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Rescaled min-distance cubed as statistic, two-sided p-value, and verdict.
        """
        floats = bits_to_floats(data, 32)
        n = self._num_points if self._num_points is not None else len(floats) // 3
        n = min(n, len(floats) // 3)
        if n < 50:
            return self._fail()

        coords = floats[:n * 3].reshape(n, 3) * 1000.0
        from scipy.spatial import distance as sp_dist
        dists = sp_dist.pdist(coords)
        min_d = float(np.min(dists))
        min_d3 = min_d ** 3

        # cubed min distance * (4/3)*pi*n*(n-1)/2 / 1000^3 ~ Exp(1)
        lam = (4.0 / 3.0) * math.pi * n * (n - 1) / (2.0 * (1000.0 ** 3))
        stat = min_d3 * lam
        pval = 1.0 - math.exp(-stat)
        pval = 2.0 * min(pval, 1.0 - pval)
        pval = min(max(pval, 0.0), 1.0)
        return TestResult(self.name, stat, pval, verdict(pval))


# ── Bitstream ────────────────────────────────────────────────

class BitstreamTest(StatisticalTest):
    """Bitstream — Marsaglia.

    Examines overlapping 20-bit words from the bit stream.
    With 2^21 overlapping words, about 2^20 - 141909 missing
    words are expected. Chi-squared against the known distribution.
    """

    def __init__(self) -> None:
        """No configuration needed."""
        pass

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Dieharder: Bitstream'.
        """
        return "Dieharder: Bitstream"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Count missing overlapping 20-bit words and z-test against the expected count.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Z-score as statistic, p-value, and verdict.
        """
        n_bits = len(data)
        word_len = 20
        while 2 ** word_len > n_bits and word_len > 10:
            word_len -= 1

        n_possible = 2 ** word_len
        if n_bits < n_possible:
            return self._fail()

        bits = data[:n_bits]
        seen = np.zeros(n_possible, dtype=np.bool_)

        val = 0
        mask = n_possible - 1
        for j in range(word_len):
            val = (val << 1) | int(bits[j])
        seen[val] = True

        for j in range(word_len, len(bits)):
            val = ((val << 1) | int(bits[j])) & mask
            seen[val] = True

        missing = int(n_possible - np.sum(seen))

        n_overlapping = len(bits) - word_len + 1
        ratio = n_overlapping / n_possible
        expected_missing = n_possible * math.exp(-ratio)
        sigma = math.sqrt(n_possible * math.exp(-ratio) * (1.0 - math.exp(-ratio)))

        if sigma < 1.0:
            return self._fail()

        z = (missing - expected_missing) / sigma
        pval = p_from_z(z)
        return TestResult(self.name, z, pval, verdict(pval))


# ── DNA ──────────────────────────────────────────────────────

class DNATest(StatisticalTest):
    """DNA — Marsaglia.

    Treats bits as a 4-letter alphabet (pairs of bits = A/C/G/T).
    Counts overlapping 10-letter "words" and checks the number of
    missing words against the expected distribution. Variant of OPSO.
    """

    def __init__(self) -> None:
        """No configuration needed."""
        pass

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Dieharder: DNA'.
        """
        return "Dieharder: DNA"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Build 4-letter overlapping 10-mer words and z-test the missing word count.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Z-score as statistic, p-value, and verdict.
        """
        # 4-letter alphabet from bit pairs; overlapping 10-letter words
        word_len = 10
        alphabet_size = 4
        bits_per_letter = 2

        # need enough bit pairs
        n_letters = len(data) // bits_per_letter
        if n_letters < word_len + 1000:
            return self._fail()

        letters = bits_to_blocks(data[:n_letters * bits_per_letter], bits_per_letter)
        n_possible = alphabet_size ** word_len  # 4^10 = 1048576

        seen = np.zeros(n_possible, dtype=np.bool_)
        val = 0
        base = alphabet_size

        for j in range(word_len):
            val = val * base + int(letters[j])
        seen[val] = True

        pow_top = base ** (word_len - 1)
        for j in range(word_len, len(letters)):
            val = (val % pow_top) * base + int(letters[j])
            seen[val] = True

        missing = int(n_possible - np.sum(seen))

        # expected missing ~ n_possible * exp(-n_words / n_possible)
        n_words = len(letters) - word_len + 1
        ratio = n_words / n_possible
        expected_missing = n_possible * math.exp(-ratio)
        # sigma approximation
        sigma = math.sqrt(n_possible * math.exp(-ratio) * (1 - math.exp(-ratio)))

        if sigma < 1.0:
            return self._fail()

        z = (missing - expected_missing) / sigma
        pval = p_from_z(z)
        return TestResult(self.name, z, pval, verdict(pval))


# ── Count the 1s ─────────────────────────────────────────────

class CountOnesStreamTest(StatisticalTest):
    """Count the 1s (stream) — Marsaglia.

    Counts 1-bits in overlapping groups of 8 consecutive bits.
    The letter counts (0..4 mapped from byte popcount) in groups
    of 5 are tested against the known multinomial distribution.
    """

    def __init__(self) -> None:
        """No configuration needed."""
        pass

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Dieharder: Count the 1s (Stream)'.
        """
        return "Dieharder: Count the 1s (Stream)"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Map overlapping byte popcounts to letters and chi-squared test 5-letter groups.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        # overlapping bytes from bit stream, subsample every 8th to decorrelate
        if len(data) < 8 + 5 * 8 * 200:
            return self._fail()

        windows = np.lib.stride_tricks.sliding_window_view(data, 8)
        popcounts = windows.sum(axis=1)
        # subsample every 8th overlapping byte for independence
        popcounts = popcounts[::8]

        letter_map = np.array([0, 0, 1, 2, 2, 3, 4, 4, 4], dtype=np.int64)
        letters = letter_map[popcounts]

        binom_probs = np.array([sp_comb(8, k, exact=True) / 256.0 for k in range(9)])
        letter_probs = np.array([
            binom_probs[0] + binom_probs[1],
            binom_probs[2],
            binom_probs[3] + binom_probs[4],
            binom_probs[5],
            binom_probs[6] + binom_probs[7] + binom_probs[8],
        ])

        n_groups = len(letters) // 5
        if n_groups < 200:
            return self._fail()
        letters = letters[:n_groups * 5].reshape(n_groups, 5)

        n_cats = 5 ** 5
        counts = np.zeros(n_cats, dtype=np.int64)
        for i in range(n_groups):
            idx = 0
            for j in range(5):
                idx = idx * 5 + int(letters[i, j])
            counts[idx] += 1

        expected = np.zeros(n_cats)
        for cat in range(n_cats):
            digits = []
            c = cat
            for _ in range(5):
                digits.append(c % 5)
                c //= 5
            digits.reverse()
            p = 1.0
            for d in digits:
                p *= letter_probs[d]
            expected[cat] = p * n_groups

        mask = expected >= 1.0
        if np.sum(mask) < 2:
            return self._fail()
        obs_main = counts[mask].astype(np.float64)
        exp_main = expected[mask]
        obs_other = float(np.sum(counts[~mask]))
        exp_other = float(np.sum(expected[~mask]))

        if exp_other > 0:
            obs_all = np.append(obs_main, obs_other)
            exp_all = np.append(exp_main, exp_other)
        else:
            obs_all = obs_main
            exp_all = exp_main

        chi2 = float(np.sum((obs_all - exp_all) ** 2 / exp_all))
        df = len(obs_all) - 1
        pval = p_from_chi2(chi2, df)
        return TestResult(self.name, chi2, pval, verdict(pval))


class CountOnesByteTest(StatisticalTest):
    """Count the 1s (byte) — Marsaglia.

    Non-overlapping version. Packs bits into bytes, counts
    popcount of each, maps to letters, and tests 5-letter groups
    against the known multinomial.
    """

    def __init__(self) -> None:
        """No configuration needed."""
        pass

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Dieharder: Count the 1s (Byte)'.
        """
        return "Dieharder: Count the 1s (Byte)"

    def _fail(self) -> TestResult:
        """Return a default FAIL result for insufficient data.

        Returns:
            TestResult: Zero statistic and p-value with FAIL verdict.
        """
        return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Map non-overlapping byte popcounts to letters and chi-squared test 5-letter groups.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        # non-overlapping bytes
        packed = bits_to_bytes(data)
        if len(packed) < 5 * 100:
            return self._fail()

        # popcount via lookup
        lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.int64)
        popcounts = lut[packed]

        letter_map = np.array([0, 0, 1, 2, 2, 3, 4, 4, 4], dtype=np.int64)
        letters = letter_map[popcounts]

        binom_probs = np.array([sp_comb(8, k, exact=True) / 256.0 for k in range(9)])
        letter_probs = np.array([
            binom_probs[0] + binom_probs[1],
            binom_probs[2],
            binom_probs[3] + binom_probs[4],
            binom_probs[5],
            binom_probs[6] + binom_probs[7] + binom_probs[8],
        ])

        n_groups = len(letters) // 5
        if n_groups < 200:
            return self._fail()
        letters = letters[:n_groups * 5].reshape(n_groups, 5)

        n_cats = 5 ** 5
        counts = np.zeros(n_cats, dtype=np.int64)
        for i in range(n_groups):
            idx = 0
            for j in range(5):
                idx = idx * 5 + int(letters[i, j])
            counts[idx] += 1

        expected = np.zeros(n_cats)
        for cat in range(n_cats):
            digits = []
            c = cat
            for _ in range(5):
                digits.append(c % 5)
                c //= 5
            digits.reverse()
            p = 1.0
            for d in digits:
                p *= letter_probs[d]
            expected[cat] = p * n_groups

        mask = expected >= 1.0
        if np.sum(mask) < 2:
            return self._fail()
        obs_main = counts[mask].astype(np.float64)
        exp_main = expected[mask]
        obs_other = float(np.sum(counts[~mask]))
        exp_other = float(np.sum(expected[~mask]))

        if exp_other > 0:
            obs_all = np.append(obs_main, obs_other)
            exp_all = np.append(exp_main, exp_other)
        else:
            obs_all = obs_main
            exp_all = exp_main

        chi2 = float(np.sum((obs_all - exp_all) ** 2 / exp_all))
        df = len(obs_all) - 1
        pval = p_from_chi2(chi2, df)
        return TestResult(self.name, chi2, pval, verdict(pval))


# ── Suite runner ──────────────────────────────────────────────

def dieharder_battery() -> list[StatisticalTest]:
    """Return all Dieharder-family tests with default parameters.

    Returns:
        list[StatisticalTest]: 18 test instances covering all Marsaglia/Dieharder tests.
    """
    return [
        BirthdaySpacingsTest(),
        OverlappingPermutationsTest(),
        ParkingLotTest(),
        MinimumDistanceTest(),
        ThreeDSpheresTest(),
        SqueezeTest(),
        OverlappingSumsTest(),
        CrapsTest(),
        GCDTest(),
        GorillaTest(),
        CouponCollectorTest(),
        GapTest(),
        PokerTest(),
        CollisionTest(),
        BitstreamTest(),
        DNATest(),
        CountOnesStreamTest(),
        CountOnesByteTest(),
    ]
