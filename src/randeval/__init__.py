"""randeval — generate and evaluate random number sequences."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .sequence import RandomSequence, SequenceMetadata
from .tests_statistical.base import TestSuiteResult

from . import generators
from . import extractors
from . import tests_statistical

from .generators import list_all as list_all_generators
from .generators import all_generators
from .extractors import list_all as list_all_extractors
from .extractors import all_extractors, default_extractors
from .tests_statistical import list_all as list_all_tests
from .tests_statistical import full_battery


def evaluate(
    data: NDArray[np.uint8] | list[int] | bytes,
    battery: str | None = None,
) -> TestSuiteResult:
    """Run a full test battery on raw bit data.

    Args:
        data: Bit array (0s and 1s), list of ints, or raw bytes.
        battery: Battery name — "nist", "dieharder", "sp800_90b",
            "entropy", "distribution", "novel", or None for full.

    Returns:
        TestSuiteResult: Aggregated results with .summary(), .to_dataframe(), etc.
    """
    seq = RandomSequence(data)
    return seq.test_all(battery)


__all__: list[str] = [
    "RandomSequence",
    "SequenceMetadata",
    "TestSuiteResult",
    "generators",
    "extractors",
    "tests_statistical",
    "list_all_generators",
    "all_generators",
    "list_all_extractors",
    "all_extractors",
    "default_extractors",
    "list_all_tests",
    "full_battery",
    "evaluate",
]
