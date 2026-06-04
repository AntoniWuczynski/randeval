"""General-purpose statistical distribution tests."""

from __future__ import annotations

from math import exp, log, sqrt

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats

from .base import StatisticalTest, TestResult, Verdict
from ._utils import bits_to_blocks, bits_to_floats, p_from_chi2, p_from_z, verdict


class ChiSquaredUniformityTest(StatisticalTest):
    """Chi-squared goodness-of-fit against uniform distribution.

    Groups bits into k-bit integers and tests whether all 2^k values
    appear with equal frequency.
    """

    def __init__(self, *, bits_per_value: int = 8) -> None:
        """Configure the number of bits per integer value.

        Args:
            bits_per_value: Bits per integer (k). Tests 2^k bins.
        """
        self._bits_per_value = bits_per_value

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'Chi-Squared Uniformity (k=8)'.
        """
        return f"Chi-Squared Uniformity (k={self._bits_per_value})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Group bits into k-bit integers and chi-squared test against uniform distribution.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Chi-squared statistic, p-value, and verdict.
        """
        blocks = bits_to_blocks(data, self._bits_per_value)
        if len(blocks) == 0:
            return TestResult(self.name, 0.0, 0.0, Verdict.FAIL)
        n_bins = 2 ** self._bits_per_value
        observed = np.bincount(blocks, minlength=n_bins).astype(np.float64)
        expected = len(blocks) / n_bins
        chi2 = float(np.sum((observed - expected) ** 2 / expected))
        df = n_bins - 1
        p = p_from_chi2(chi2, df)
        return TestResult(test_name=self.name, statistic=chi2, p_value=p, verdict=verdict(p))


class KolmogorovSmirnovTest(StatisticalTest):
    """Kolmogorov-Smirnov test against Uniform(0,1).

    Converts bit blocks to floats in [0,1) and computes the KS
    statistic against the theoretical CDF.
    """

    def __init__(self, *, bits_per_value: int = 32) -> None:
        """Configure bits per float value for the KS test.

        Args:
            bits_per_value: Number of bits per float value (k).
        """
        self._bits_per_value = bits_per_value

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'Kolmogorov-Smirnov (k=32)'.
        """
        return f"Kolmogorov-Smirnov (k={self._bits_per_value})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Convert bits to floats and KS-test against the uniform CDF.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: KS statistic, p-value, and verdict.
        """
        values = bits_to_floats(data, self._bits_per_value)
        stat, p = sp_stats.kstest(values, "uniform")
        return TestResult(test_name=self.name, statistic=float(stat), p_value=float(p), verdict=verdict(float(p)))


class AndersonDarlingTest(StatisticalTest):
    """Anderson-Darling test against Uniform(0,1).

    More sensitive to tail deviations than KS. Converts bit blocks
    to floats and tests against uniform CDF.
    """

    def __init__(self, *, bits_per_value: int = 32) -> None:
        """Configure bits per float value for the Anderson-Darling test.

        Args:
            bits_per_value: Number of bits per float value (k).
        """
        self._bits_per_value = bits_per_value

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'Anderson-Darling (k=32)'.
        """
        return f"Anderson-Darling (k={self._bits_per_value})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Convert bits to floats and compute the AD statistic against uniform CDF.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: A-squared statistic, approximate p-value, and verdict.
        """
        values = bits_to_floats(data, self._bits_per_value)
        n = len(values)
        u = np.sort(values)
        # clamp to avoid log(0)
        u = np.clip(u, 1e-15, 1.0 - 1e-15)
        idx = np.arange(1, n + 1)
        a2 = -n - (1.0 / n) * np.sum((2 * idx - 1) * (np.log(u) + np.log(1.0 - u[::-1])))
        a2 = float(a2)
        # asymptotic p-value for uniform AD (Marsaglia & Marsaglia 2004 approximation)
        p = self._ad_p_value(a2)
        return TestResult(test_name=self.name, statistic=a2, p_value=p, verdict=verdict(p))

    @staticmethod
    def _ad_p_value(a2: float) -> float:
        """Approximate p-value for the AD statistic against uniform(0,1).

        Args:
            a2: Anderson-Darling A-squared statistic.

        Returns:
            float: Approximate p-value via the Marsaglia & Marsaglia (2004) formula.
        """
        if a2 <= 0.0:
            return 1.0
        # use scipy's goodness-of-fit machinery: transform to exponential and use known table
        # Simpler: use the Marsaglia formula for the case-specific distribution
        # For AD uniform, we use the modified statistic and kstwobign as fallback
        # Direct approximation from Marsaglia & Marsaglia (2004)
        if a2 < 2.0:
            p = 1.0 - a2 ** (-0.5) * exp(
                -1.2337141 / a2
                + 0.6187463 * log(a2)
                - 0.1043243
                - 0.1163606 / a2
                + 0.0128576 / (a2 * a2)
            ) if a2 >= 0.2 else 1.0 - exp(-13.436 + 101.14 * a2 - 223.73 * a2 ** 2)
        else:
            p = exp(
                -0.4938691 * a2
                - 0.2325035
                + 0.1075702 / a2
                - 0.0024998 / (a2 * a2)
            ) if a2 < 6.0 else exp(-1.0776 * a2 + 0.4443)
        return max(0.0, min(1.0, p))


class WaldWolfowitzRunsTest(StatisticalTest):
    """Wald-Wolfowitz runs test on the bit sequence.

    A general non-parametric test for independence. Counts runs
    above/below the median value.
    """

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Wald-Wolfowitz Runs'.
        """
        return "Wald-Wolfowitz Runs"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Count runs of consecutive 0s/1s and z-test against the expected count.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Z-score as statistic, p-value, and verdict.
        """
        n = len(data)
        n1 = int(data.sum())
        n0 = n - n1
        if n0 == 0 or n1 == 0:
            return TestResult(test_name=self.name, statistic=0.0, p_value=0.0, verdict=verdict(0.0))
        # count runs: a run starts whenever the bit changes
        runs = 1 + int(np.sum(data[1:] != data[:-1]))
        e_r = 1.0 + 2.0 * n0 * n1 / n
        var_r = 2.0 * n0 * n1 * (2.0 * n0 * n1 - n) / (n * n * (n - 1.0))
        if var_r <= 0:
            return TestResult(test_name=self.name, statistic=0.0, p_value=1.0, verdict=verdict(1.0))
        z = (runs - e_r) / sqrt(var_r)
        p = p_from_z(z)
        return TestResult(test_name=self.name, statistic=float(z), p_value=p, verdict=verdict(p))


class MannKendallTrendTest(StatisticalTest):
    """Mann-Kendall trend test.

    Detects monotonic trends in the sequence. A truly random
    sequence should show no significant trend.
    """

    def __init__(self, *, block_size: int = 8) -> None:
        """Configure the block size for grouping bits into comparable values.

        Args:
            block_size: Number of bits per block for integer conversion.
        """
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Mann-Kendall Trend'.
        """
        return "Mann-Kendall Trend"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Compute the Mann-Kendall S statistic and z-test for monotonic trend.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Z-score as statistic, p-value, and verdict.
        """
        # group into blocks for meaningful comparisons
        blocks = bits_to_blocks(data, self._block_size)
        n = len(blocks)
        # vectorised S computation: count all concordant - discordant pairs
        s = 0
        # for large n, process in chunks to avoid huge memory allocation
        for i in range(n - 1):
            diffs = blocks[i + 1:].astype(np.int64) - int(blocks[i])
            s += int(np.sum(np.sign(diffs)))
        _, tie_counts = np.unique(blocks, return_counts=True)
        tied_groups = tie_counts[tie_counts > 1]
        tie_correction = sum(t * (t - 1) * (2 * t + 5) for t in tied_groups)
        var_s = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18.0
        z = s / sqrt(var_s) if var_s > 0 else 0.0
        p = p_from_z(z)
        return TestResult(test_name=self.name, statistic=float(z), p_value=p, verdict=verdict(p))


class TurningPointTest(StatisticalTest):
    """Turning Point test.

    Counts local maxima and minima. For n random values, the expected
    number of turning points is 2(n-2)/3 with known variance.
    """

    def __init__(self, *, block_size: int = 8) -> None:
        """Configure the block size for grouping bits into comparable values.

        Args:
            block_size: Number of bits per block for integer conversion.
        """
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Turning Point'.
        """
        return "Turning Point"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Count local maxima/minima and z-test against the expected 2(n-2)/3.

        Adds small jitter to break ties from discrete block values,
        since the theoretical formula assumes continuous data with
        probability-zero ties.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Z-score as statistic, p-value, and verdict.
        """
        blocks = bits_to_blocks(data, self._block_size)
        n = len(blocks)
        if n < 3:
            return TestResult(test_name=self.name, statistic=0.0, p_value=1.0, verdict=verdict(1.0))
        # break ties with deterministic jitter so the continuous-data formula applies
        values = blocks.astype(np.float64) + np.random.default_rng(0).uniform(-0.5, 0.5, n)
        # a turning point at i if values[i] is a local max or min
        tp = int(np.sum(
            ((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))
            | ((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:]))
        ))
        expected = 2.0 * (n - 2) / 3.0
        var = (16.0 * n - 29.0) / 90.0
        z = (tp - expected) / sqrt(var)
        p = p_from_z(z)
        return TestResult(test_name=self.name, statistic=float(z), p_value=p, verdict=verdict(p))


def distribution_battery() -> list[StatisticalTest]:
    """Return all distribution tests with default parameters.

    Returns:
        list[StatisticalTest]: 6 distribution test instances.
    """
    return [
        ChiSquaredUniformityTest(),
        KolmogorovSmirnovTest(),
        AndersonDarlingTest(),
        WaldWolfowitzRunsTest(),
        MannKendallTrendTest(),
        TurningPointTest(),
    ]
