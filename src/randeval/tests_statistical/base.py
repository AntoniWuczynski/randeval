from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd


class Verdict(Enum):
    """Pass/fail outcome of a statistical test."""
    PASS = "pass"
    FAIL = "fail"


@dataclass(frozen=True)
class TestResult:
    """Immutable result of a single statistical test."""

    test_name: str
    statistic: float
    p_value: float
    verdict: Verdict

    def __repr__(self) -> str:
        """Return a compact string showing name, p-value, and verdict.

        Returns:
            str: Like "TestResult('Frequency', p=0.1234, pass)".
        """
        return (
            f"TestResult({self.test_name!r}, p={self.p_value:.4f}, "
            f"{self.verdict.value})"
        )


class TestSuiteResult:
    """Wrapper around a list of TestResults with aggregation helpers."""

    def __init__(self, results: list[TestResult], source: str | None = None) -> None:
        """Wrap a list of TestResults with an optional source label.

        Args:
            results: Individual test outcomes to aggregate.
            source: Human-readable description of the tested sequence.
        """
        self._results = results
        self._source = source

    @property
    def results(self) -> list[TestResult]:
        """All individual test results.

        Returns:
            list[TestResult]: The wrapped result list.
        """
        return self._results

    @property
    def source(self) -> str | None:
        """Description of the sequence that was tested, if available.

        Returns:
            str | None: Source label, or None if not set.
        """
        return self._source

    def __len__(self) -> int:
        """Return number of test results.

        Returns:
            int: Count of results in the suite.
        """
        return len(self._results)

    def __iter__(self) -> Iterator[TestResult]:
        """Iterate over individual TestResult objects.

        Returns:
            Iterator[TestResult]: Iterator over the results list.
        """
        return iter(self._results)

    def __getitem__(self, idx: int | slice) -> TestResult | list[TestResult]:
        """Index or slice into the results list.

        Args:
            idx: Integer index or slice.

        Returns:
            TestResult | list[TestResult]: Single result or sub-list.
        """
        return self._results[idx]

    @property
    def passes(self) -> list[TestResult]:
        """Results that passed (p >= alpha).

        Returns:
            list[TestResult]: Only the passing results.
        """
        return [r for r in self._results if r.verdict == Verdict.PASS]

    @property
    def failures(self) -> list[TestResult]:
        """Results that failed (p < alpha).

        Returns:
            list[TestResult]: Only the failing results.
        """
        return [r for r in self._results if r.verdict == Verdict.FAIL]

    @property
    def pass_rate(self) -> float:
        """Fraction of tests that passed, 0.0 if no results.

        Returns:
            float: Pass rate between 0.0 and 1.0.
        """
        if not self._results:
            return 0.0
        return len(self.passes) / len(self._results)

    def filter(self, verdict: Verdict | None = None) -> list[TestResult]:
        """Return results matching the given verdict, or all if None.

        Args:
            verdict: Filter to PASS or FAIL, or None for everything.

        Returns:
            list[TestResult]: Filtered (or full) copy of the results list.
        """
        if verdict is None:
            return list(self._results)
        return [r for r in self._results if r.verdict == verdict]

    def to_dict(self) -> list[dict[str, object]]:
        """Serialize results to a list of plain dicts.

        Returns:
            list[dict[str, object]]: Each dict has keys: test, statistic, p_value, verdict.
        """
        return [
            {
                "test": r.test_name,
                "statistic": r.statistic,
                "p_value": r.p_value,
                "verdict": r.verdict.value,
            }
            for r in self._results
        ]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame.

        Returns:
            pd.DataFrame: Columns: test, statistic, p_value, verdict.

        Raises:
            ImportError: If pandas is not installed.
        """
        import pandas as pd
        return pd.DataFrame(self.to_dict())

    def summary(self) -> str:
        """Format a human-readable table of all results with pass/fail counts.

        Returns:
            str: Multi-line table string with header, rows, and totals.
        """
        lines = [
            f"{'Test':50s}  {'Statistic':>10s}  {'p-value':>8s}  {'Verdict':>7s}",
            "-" * 80,
        ]
        for r in self._results:
            v = "PASS" if r.verdict == Verdict.PASS else "FAIL"
            lines.append(f"{r.test_name:50s}  {r.statistic:10.4f}  {r.p_value:8.4f}  {v:>7s}")
        lines.append("-" * 80)
        lines.append(
            f"Total: {len(self._results)} tests | "
            f"{len(self.passes)} passed | "
            f"{len(self.failures)} failed | "
            f"pass rate: {self.pass_rate:.1%}"
        )
        if self._source:
            lines.append(f"Source: {self._source}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a compact string showing test count and pass rate.

        Returns:
            str: Like "TestSuiteResult(16 tests, pass_rate=87.5%)".
        """
        return (
            f"TestSuiteResult({len(self._results)} tests, "
            f"pass_rate={self.pass_rate:.1%})"
        )


class StatisticalTest(ABC):
    """Abstract base for statistical randomness tests."""

    @abstractmethod
    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Run the test on a bit array and return the result.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Contains test statistic, p-value, and verdict.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this test.

        Returns:
            str: Display name like 'NIST 1: Frequency (Monobit)'.
        """
        ...
