"""Novel statistical tests that don't appear in NIST, Dieharder, or TestU01.

These are what make randeval unique — meta-analysis, spatial structure,
and distribution-level tests that standard suites skip.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from .base import StatisticalTest, TestResult, TestSuiteResult, Verdict
from ._utils import bits_to_blocks, bits_to_floats, bits_to_bytes, p_from_z, p_from_chi2, verdict


# ── P-Value Uniformity (meta-test) ──────────────────────────────

class PValueUniformityTest(StatisticalTest):
    """Tests whether p-values from a test suite are uniform on [0,1].

    No standard framework checks its own calibration. This one does.
    If suite_results is given at construction, run() ignores data and
    tests those p-values directly. Otherwise it runs the full battery first.
    """

    def __init__(self, *, suite_results: TestSuiteResult | None = None) -> None:
        """Optionally pass pre-computed suite results; otherwise run() runs the full battery.

        Args:
            suite_results: Pre-computed results to extract p-values from. If None,
                run() will execute the full battery first.
        """
        self._suite_results = suite_results

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Novel: P-Value Uniformity'.
        """
        return "Novel: P-Value Uniformity"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Collect p-values from test results and KS-test them against Uniform(0,1).

        Args:
            data: 1-D array of 0s and 1s (ignored if suite_results was provided).

        Returns:
            TestResult: KS statistic, p-value from the uniformity check, and verdict.
        """
        if self._suite_results is not None:
            pvals = [r.p_value for r in self._suite_results]
        else:
            from . import full_battery
            from .base import TestSuiteResult as TSR
            tests = full_battery()
            results = [t.run(data) for t in tests]
            pvals = [r.p_value for r in results]

        # keep only valid p-values in [0, 1]
        pvals = [p for p in pvals if 0.0 <= p <= 1.0]

        if len(pvals) < 3:
            return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

        ks_stat, ks_pval = sp_stats.kstest(pvals, "uniform")
        return TestResult(self.name, float(ks_stat), float(ks_pval), verdict(ks_pval))


# ── Running Bias ────────────────────────────────────────────────

class RunningBiasTest(StatisticalTest):
    """Sliding-window bias detector.

    Global frequency tests average everything out. This catches
    local drift — a window where the generator was temporarily biased.
    """

    def __init__(self, *, window_size: int = 1000) -> None:
        """Configure the sliding window width for local bias detection.

        Args:
            window_size: Number of bits per window.
        """
        self._window_size = window_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Novel: Running Bias'.
        """
        return "Novel: Running Bias"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Find the window with the largest deviation from 0.5 mean and Bonferroni-test it.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Max window deviation as statistic, Bonferroni p-value, and verdict.
        """
        w = self._window_size
        n = len(data)
        if n < w:
            return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

        cumsum = np.cumsum(data.astype(np.float64))
        # running mean over each window
        window_sums = cumsum[w - 1:] - np.concatenate([[0.0], cumsum[:n - w]])
        window_means = window_sums / w
        devs = np.abs(window_means - 0.5)
        max_dev = float(np.max(devs))
        num_windows = len(devs)

        z = max_dev * math.sqrt(w) * 2.0
        # bonferroni correction for multiple windows
        raw_p = p_from_z(z)
        pval = min(raw_p * num_windows, 1.0)

        return TestResult(self.name, max_dev, pval, verdict(pval))


# ── Bit Pattern Spatial ─────────────────────────────────────────

class BitPatternSpatialTest(StatisticalTest):
    """Reshape bits onto a 2D grid, check block-level density uniformity.

    Physical RNGs can have spatial correlations if the source has
    geometric structure (e.g. camera-based QRNGs). Standard tests
    treat the stream as purely sequential.
    """

    def __init__(self, *, block_rows: int = 8, block_cols: int = 8) -> None:
        """Configure 2D block dimensions for spatial density analysis.

        Args:
            block_rows: Height of each spatial block in bits.
            block_cols: Width of each spatial block in bits.
        """
        self._brows = block_rows
        self._bcols = block_cols

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Novel: Bit Pattern Spatial'.
        """
        return "Novel: Bit Pattern Spatial"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Reshape bits onto a 2D grid and chi-squared test block-level density uniformity.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        br, bc = self._brows, self._bcols
        block_area = br * bc

        # figure out grid dimensions that fit the data
        total = len(data)
        # pick grid width as a multiple of bc
        grid_cols = bc * max(1, int(math.sqrt(total / (br / bc))))
        grid_rows = total // grid_cols
        # trim to fit whole blocks
        grid_rows = (grid_rows // br) * br
        grid_cols = (grid_cols // bc) * bc

        used = grid_rows * grid_cols
        if used < block_area or grid_rows < br or grid_cols < bc:
            return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

        grid = data[:used].reshape(grid_rows, grid_cols)

        nblocks_r = grid_rows // br
        nblocks_c = grid_cols // bc
        nblocks = nblocks_r * nblocks_c

        # count ones in each block
        counts = np.zeros(nblocks, dtype=np.float64)
        idx = 0
        for i in range(nblocks_r):
            for j in range(nblocks_c):
                block = grid[i * br:(i + 1) * br, j * bc:(j + 1) * bc]
                counts[idx] = block.sum()
                idx += 1

        expected = block_area * 0.5
        variance = block_area * 0.25
        chi2 = float(np.sum((counts - expected) ** 2 / variance))
        df = nblocks - 1
        if df < 1:
            return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

        pval = p_from_chi2(chi2, df)
        return TestResult(self.name, chi2, pval, verdict(pval))


# ── Weight Distribution ─────────────────────────────────────────

class WeightDistributionTest(StatisticalTest):
    """Hamming weight of each block should follow Binomial(block_size, 0.5).

    Inspired by TestU01's "Weight" family. Catches generators where
    individual bytes are well-distributed but bit-level balance is off.
    """

    def __init__(self, *, block_size: int = 8) -> None:
        """Configure the block size for Hamming weight analysis.

        Args:
            block_size: Number of bits per block.
        """
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Novel: Weight Distribution'.
        """
        return "Novel: Weight Distribution"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Compute Hamming weights per block and chi-squared test against Binomial(bs, 0.5).

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        bs = self._block_size
        n = len(data) - (len(data) % bs)
        if n < bs * 10:
            return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

        blocks = data[:n].reshape(-1, bs)
        weights = blocks.sum(axis=1)
        nblocks = len(weights)

        # expected counts for each possible weight 0..bs
        observed = np.bincount(weights, minlength=bs + 1).astype(np.float64)
        expected = np.array([
            sp_stats.binom.pmf(k, bs, 0.5) * nblocks
            for k in range(bs + 1)
        ])

        # merge bins with expected < 5
        obs_merged: list[float] = []
        exp_merged: list[float] = []
        o_acc, e_acc = 0.0, 0.0
        for o, e in zip(observed, expected):
            o_acc += o
            e_acc += e
            if e_acc >= 5.0:
                obs_merged.append(o_acc)
                exp_merged.append(e_acc)
                o_acc, e_acc = 0.0, 0.0
        if e_acc > 0:
            if exp_merged:
                obs_merged[-1] += o_acc
                exp_merged[-1] += e_acc
            else:
                obs_merged.append(o_acc)
                exp_merged.append(e_acc)

        obs_arr = np.array(obs_merged)
        exp_arr = np.array(exp_merged)
        df = len(obs_arr) - 1
        if df < 1:
            return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

        chi2 = float(np.sum((obs_arr - exp_arr) ** 2 / exp_arr))
        pval = p_from_chi2(chi2, df)
        return TestResult(self.name, chi2, pval, verdict(pval))


# ── Close Pairs ─────────────────────────────────────────────────

class ClosePairsTest(StatisticalTest):
    """Count of close point pairs in d-dimensional space.

    Place n points in [0,1)^d and count pairs within a small radius using
    toroidal (wrap-around) distance. Under uniformity that count is Poisson;
    a generator that clusters or repels shifts it. Toroidal distance avoids
    boundary bias, and counting (one statistic) avoids the dependence problem
    of per-point nearest-neighbour distances.
    """

    def __init__(self, *, dimensions: int = 2, num_points: int | None = None) -> None:
        """Configure dimensionality and optional point count.

        Args:
            dimensions: Number of spatial dimensions (d).
            num_points: Number of points to place. Auto-derived from data if None.
        """
        self._dim = dimensions
        self._num_points = num_points

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Novel: Close Pairs'.
        """
        return "Novel: Close Pairs"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Count point pairs within a small radius and test the count against Poisson.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: close-pair count as statistic, two-sided Poisson p-value, and verdict.
        """
        d = self._dim
        floats = bits_to_floats(data, 32)
        max_pts = len(floats) // d
        n = self._num_points if self._num_points is not None else min(max_pts, 2000)
        n = min(n, max_pts)

        if n < 50:
            return TestResult(self.name, 0.0, 1.0, Verdict.PASS)

        pts = floats[:n * d].reshape(n, d)
        v_d = math.pi ** (d / 2.0) / math.gamma(d / 2.0 + 1.0)
        n_pairs = n * (n - 1) / 2.0

        # radius chosen so we expect ~20 close pairs under uniformity: small
        # enough that the count is Poisson (Chen-Stein), big enough for power
        lam = 20.0
        r = (lam / (n_pairs * v_d)) ** (1.0 / d)

        # toroidal squared distances (per dim, to keep memory modest)
        d2 = np.zeros((n, n))
        for k in range(d):
            delta = np.abs(pts[:, k][:, None] - pts[:, k][None, :])
            delta = np.minimum(delta, 1.0 - delta)
            d2 += delta ** 2
        count = int((d2 < r * r).sum() - n) // 2

        p_left = float(sp_stats.poisson.cdf(count, lam))
        p_right = float(sp_stats.poisson.sf(count - 1, lam))
        pval = min(1.0, 2.0 * min(p_left, p_right))
        return TestResult(self.name, float(count), pval, verdict(pval))


# ── Max-of-t ───────────────────────────────────────────────────

class MaxOfTTest(StatisticalTest):
    """Max of t uniform values should have CDF x^t.

    Simple but effective — catches generators with reduced effective range.
    """

    def __init__(self, *, t: int = 5) -> None:
        """Configure how many uniform values to take the max of per group.

        Args:
            t: Group size -- max of t values per observation.
        """
        self._t = t

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Novel: Max-of-t'.
        """
        return "Novel: Max-of-t"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Take the max of each group of t floats and KS-test against CDF x^t.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: KS statistic, p-value, and verdict.
        """
        t = self._t
        floats = bits_to_floats(data, 32)
        ngroups = len(floats) // t
        if ngroups < 20:
            return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

        grouped = floats[:ngroups * t].reshape(ngroups, t)
        maxvals = grouped.max(axis=1)

        ks_stat, ks_pval = sp_stats.kstest(maxvals, lambda x: x ** t)
        return TestResult(self.name, float(ks_stat), float(ks_pval), verdict(ks_pval))


# ── Successive Differences ──────────────────────────────────────

class SuccessiveDifferenceTest(StatisticalTest):
    """Mean absolute successive difference of block-level integers.

    For uniform values in [0, M), E[|d_i|] = M/3. Deviations indicate
    sequential correlation between consecutive blocks.
    """

    def __init__(self, *, block_size: int = 8) -> None:
        """Configure the block size for integer-level differencing.

        Args:
            block_size: Number of bits per block.
        """
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Novel: Successive Differences'.
        """
        return "Novel: Successive Differences"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Compute mean |x_i - x_{i+1}| over blocks and z-test against E[|X-Y|] = M/3.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Mean absolute difference as statistic, z-based p-value, and verdict.
        """
        bs = self._block_size
        blocks = bits_to_blocks(data, bs)
        n = len(blocks)
        if n < 30:
            return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

        M = 2 ** bs
        diffs = np.abs(np.diff(blocks.astype(np.float64)))
        mean_diff = float(np.mean(diffs))

        # E[|X-Y|] = M/3 for independent uniform on {0,...,M-1}
        expected = M / 3.0
        # Var[|X-Y|] is (M^2/18)(4 - 1/(something)) but approximate:
        # Var[|X-Y|] ~ M^2 * (1/2 - 1/9) = M^2 * 7/18 for continuous
        var_single = M ** 2 * 7.0 / 18.0
        ndiffs = len(diffs)
        se = math.sqrt(var_single / ndiffs)

        z = (mean_diff - expected) / se if se > 0 else 0.0
        pval = p_from_z(z)
        return TestResult(self.name, mean_diff, pval, verdict(pval))


# ── Byte Runs ───────────────────────────────────────────────────

class ByteRunsTest(StatisticalTest):
    """Runs of identical block values.

    Like the classical runs test, but on block-level symbols instead
    of individual bits. With alphabet size k = 2^block_size, most
    consecutive values should differ.
    """

    def __init__(self, *, block_size: int = 8) -> None:
        """Configure the block size for symbol-level run counting.

        Args:
            block_size: Number of bits per symbol.
        """
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Novel: Byte Runs'.
        """
        return "Novel: Byte Runs"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Count runs of identical block values and z-test against the expected count.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Run count as statistic, z-based p-value, and verdict.
        """
        bs = self._block_size
        blocks = bits_to_blocks(data, bs)
        n = len(blocks)
        if n < 30:
            return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)

        k = 2 ** bs
        # count runs: number of times value changes, plus 1
        changes = np.sum(blocks[1:] != blocks[:-1])
        runs = int(changes) + 1

        # E[R] = 1 + (n-1)(1 - 1/k)
        er = 1.0 + (n - 1) * (1.0 - 1.0 / k)
        # Var[R] = (n-1) * (1 - 1/k) * (1 - (2*n - 3)/(k*(n-1)) + (n-2)/k^2)
        # simplified for large k: Var ≈ (n-1)(1-1/k)(1/k + 1 - 2/k)
        # exact: from Wald-Wolfowitz on k symbols
        p = 1.0 - 1.0 / k  # prob consecutive values differ
        vr = (n - 1) * p * (1.0 - p)
        # this is variance of the number of changes (binomial-ish), so var of runs
        if vr <= 0:
            return TestResult(self.name, 0.0, 1.0, Verdict.PASS)

        z = (runs - er) / math.sqrt(vr)
        pval = p_from_z(z)
        return TestResult(self.name, float(runs), pval, verdict(pval))


# ── Battery ─────────────────────────────────────────────────────

def novel_battery() -> list[StatisticalTest]:
    """Return all novel tests (PValueUniformityTest excluded -- it's a meta-test).

    Returns:
        list[StatisticalTest]: 7 novel test instances.
    """
    return [
        RunningBiasTest(),
        BitPatternSpatialTest(),
        WeightDistributionTest(),
        ClosePairsTest(),
        MaxOfTTest(),
        SuccessiveDifferenceTest(),
        ByteRunsTest(),
    ]
