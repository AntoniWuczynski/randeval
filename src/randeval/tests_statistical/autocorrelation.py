from __future__ import annotations

from math import erfc, sqrt

import numpy as np
from numpy.typing import NDArray

from .base import StatisticalTest, TestResult
from ._utils import verdict


class AutocorrelationTest(StatisticalTest):
    """Autocorrelation test at specified lags.

    Checks for sequential dependence between bits separated by each lag.
    """

    def __init__(self, *, lags: list[int] | None = None) -> None:
        """Configure which lags to test.

        Args:
            lags: List of integer lag values to check (default 1..100).
        """
        self._lags = lags or list(range(1, 101))

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'Autocorrelation'.
        """
        return "Autocorrelation"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Compute autocorrelation at each lag and Bonferroni-correct the best p-value.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Max |z| across lags and the Bonferroni-corrected p-value.
        """
        n = len(data)
        x = 2 * data.astype(np.float64) - 1

        valid_lags = [d for d in self._lags if d < n - 1]
        if len(valid_lags) == 0:
            return TestResult(
                test_name=self.name, statistic=0.0, p_value=0.5, verdict=verdict(0.5)
            )
        lags = np.array(valid_lags, dtype=np.int64)
        z_scores = np.empty(len(lags), dtype=np.float64)

        for i, d in enumerate(lags):
            r_d = np.dot(x[:n - d], x[d:]) / (n - d)
            z_scores[i] = r_d * sqrt(n - d)

        abs_z = np.abs(z_scores)
        # bonferroni: take the smallest per-lag p-value and multiply by num lags
        p_per_lag = np.array([erfc(az / sqrt(2.0)) for az in abs_z])
        num_lags = len(lags)
        p_val = min(1.0, float(np.min(p_per_lag)) * num_lags)

        stat = float(np.max(abs_z))
        return TestResult(
            test_name=self.name,
            statistic=stat,
            p_value=p_val,
            verdict=verdict(p_val),
        )
