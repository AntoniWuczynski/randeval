from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self, cast

import numpy as np
from numpy.typing import NDArray

from .extractors.base import Extractor
from .generators.base import Generator
from .tests_statistical.base import StatisticalTest, TestResult, TestSuiteResult, Verdict


@dataclass
class SequenceMetadata:
    """Provenance information attached to a RandomSequence.

    Attributes:
        generator_name: Name of the generator that produced the bits, or None.
        extractor_name: Name of the extractor applied, or None.
        original_length: Bit count before extraction, or None.
        extra: Arbitrary key-value pairs for additional context.
    """

    generator_name: str | None = None
    extractor_name: str | None = None
    original_length: int | None = None
    extra: dict[str, str] = field(default_factory=dict)


class RandomSequence:
    """Core object representing a sequence of random bits.

    Construct from raw data or generate via a Generator.
    """

    _data: NDArray[np.uint8]
    _metadata: SequenceMetadata

    def __init__(
        self,
        data: NDArray[np.uint8] | list[int] | bytes,
        *,
        metadata: SequenceMetadata | None = None,
    ) -> None:
        """Build a RandomSequence from raw bit data.

        Args:
            data: Input bits as a numpy array of 0s/1s, a plain list, or
                packed bytes (unpacked automatically).
            metadata: Optional provenance info. Defaults to an empty
                SequenceMetadata if not supplied.

        Raises:
            ValueError: If the array isn't 1-D or contains values other
                than 0 and 1.
        """
        if isinstance(data, bytes):
            arr = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        elif isinstance(data, list):
            arr = np.asarray(data, dtype=np.uint8)
        else:
            arr = data

        if arr.ndim != 1:
            raise ValueError(f"Expected 1-D bit array, got shape {arr.shape}")
        if not np.isin(arr, [0, 1]).all():
            raise ValueError("Data must contain only 0s and 1s")

        self._data = arr
        self._metadata = metadata or SequenceMetadata()

    @classmethod
    def from_generator(cls, generator: Generator, n: int) -> Self:
        """Generate n random bits using a Generator.

        Args:
            generator: Any Generator instance (e.g. QRNG, SystemRandom).
            n: Number of bits to generate.

        Returns:
            RandomSequence: Fresh sequence with generator metadata attached.
        """
        bits = generator.generate(n)
        meta = SequenceMetadata(generator_name=generator.name, original_length=n)
        return cls(bits, metadata=meta)

    @property
    def data(self) -> NDArray[np.uint8]:
        """Raw bit array (0s and 1s).

        Returns:
            NDArray[np.uint8]: The underlying 1-D numpy array of bits.
        """
        return self._data

    @property
    def metadata(self) -> SequenceMetadata:
        """Provenance info for this sequence.

        Returns:
            SequenceMetadata: Generator name, extractor name, original length, etc.
        """
        return self._metadata

    def __len__(self) -> int:
        """Return the number of bits in the sequence."""
        return int(self._data.shape[0])

    def __repr__(self) -> str:
        """Show length and source name for quick inspection."""
        src = self._metadata.generator_name or "external"
        return f"RandomSequence(n={len(self)}, source={src!r})"

    # ── Operations ──────────────────────────────────────────────

    def extract(self, extractor: Extractor) -> RandomSequence:
        """Apply a randomness extractor, returning a new (shorter) sequence.

        Args:
            extractor: An Extractor instance (e.g. VonNeumannExtractor).

        Returns:
            RandomSequence: New sequence with extracted bits and updated metadata.
        """
        extracted = extractor.extract(self._data)
        meta = SequenceMetadata(
            generator_name=self._metadata.generator_name,
            extractor_name=extractor.name,
            original_length=len(self),
        )
        return RandomSequence(extracted, metadata=meta)

    def extract_all(
        self, extractors: list[Extractor] | None = None
    ) -> dict[str, "RandomSequence"]:
        """Apply every extractor and return a dict of name -> extracted sequence.

        Args:
            extractors: List of Extractor instances. Defaults to
                default_extractors() if None.

        Returns:
            dict[str, RandomSequence]: Mapping of extractor name to extracted sequence.
        """
        if extractors is None:
            from .extractors import default_extractors
            extractors = default_extractors()
        return {ext.name: self.extract(ext) for ext in extractors}

    def test(self, test: StatisticalTest) -> TestResult:
        """Run a single statistical test on this sequence.

        Args:
            test: A StatisticalTest instance to execute.

        Returns:
            TestResult: p-value, verdict, and test metadata.
        """
        return test.run(self._data)

    def test_suite(self, tests: list[StatisticalTest]) -> TestSuiteResult:
        """Run multiple statistical tests and collect results.

        Args:
            tests: List of StatisticalTest instances to run.

        Returns:
            TestSuiteResult: Combined results for all tests in the list.
        """
        results = [t.run(self._data) for t in tests]
        return TestSuiteResult(results, source=repr(self))

    def test_all(self, battery: str | None = None) -> TestSuiteResult:
        """Run a named test battery (or the full one by default).

        Args:
            battery: One of "nist", "dieharder", "sp800_90b", "entropy",
                "distribution", "novel", "full", or None for the full battery.

        Returns:
            TestSuiteResult: Aggregated results for every test in the battery.

        Raises:
            ValueError: If the battery name isn't recognised.
        """
        from .tests_statistical import (
            full_battery, nist_battery, dieharder_battery,
            sp800_90b_battery, entropy_battery, distribution_battery,
            novel_battery,
        )

        batteries = {
            None: full_battery,
            "full": full_battery,
            "nist": nist_battery,
            "dieharder": dieharder_battery,
            "sp800_90b": sp800_90b_battery,
            "entropy": entropy_battery,
            "distribution": distribution_battery,
            "novel": novel_battery,
        }

        if battery not in batteries:
            raise ValueError(
                f"Unknown battery {battery!r}. "
                f"Choose from: {list(batteries.keys())}"
            )

        tests = batteries[battery]()
        return self.test_suite(tests)

    def compare(
        self,
        other: RandomSequence,
        tests: list[StatisticalTest] | None = None,
    ) -> tuple[TestSuiteResult, TestSuiteResult]:
        """Run the same tests on both sequences and return paired results.

        Args:
            other: Another RandomSequence to compare against.
            tests: Tests to run on both. Defaults to full_battery() if None.

        Returns:
            tuple[TestSuiteResult, TestSuiteResult]: Results for self, then other.
        """
        if tests is None:
            from .tests_statistical import full_battery
            tests = full_battery()

        results_self = self.test_suite(tests)
        results_other = other.test_suite(tests)
        return results_self, results_other

    # ── Block-level views ───────────────────────────────────────

    def as_bytes(self) -> NDArray[np.uint8]:
        """Pack bits into bytes (zero-padded if length isn't a multiple of 8).

        Returns:
            NDArray[np.uint8]: Packed byte array.
        """
        padded = np.pad(self._data, (0, (-len(self._data)) % 8), constant_values=0)
        return np.packbits(padded)

    def as_blocks(self, block_size: int = 8) -> NDArray[np.int64]:
        """Pack bits into integer blocks of a given size. Trailing bits are truncated.

        Args:
            block_size: Number of bits per block. Defaults to 8.

        Returns:
            NDArray[np.int64]: Array of integers, one per block.
        """
        n = len(self._data) - (len(self._data) % block_size)
        trimmed = self._data[:n].reshape(-1, block_size)
        powers = (2 ** np.arange(block_size - 1, -1, -1)).astype(np.int64)
        return cast("NDArray[np.int64]", (trimmed * powers).sum(axis=1))

    def as_floats(self, bits_per_value: int = 32) -> NDArray[np.float64]:
        """Convert bit blocks to floats in [0, 1).

        Args:
            bits_per_value: Bits used per float. Higher means more precision.
                Defaults to 32.

        Returns:
            NDArray[np.float64]: Array of floats uniformly distributed in [0, 1).

        Raises:
            ValueError: If the sequence has fewer bits than bits_per_value.
        """
        if len(self._data) < bits_per_value:
            raise ValueError(
                f"Need at least {bits_per_value} bits for as_floats(), got {len(self._data)}"
            )
        blocks = self.as_blocks(bits_per_value)
        return cast("NDArray[np.float64]", blocks.astype(np.float64) / (2 ** bits_per_value))
