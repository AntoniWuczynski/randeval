from __future__ import annotations
import numpy as np
import pytest
from randeval.extractors.practical import (
    XORExtractor, BlockParityExtractor, BitDecimationExtractor,
    WindowedXORExtractor, RepetitionFilter, ModularReductionExtractor,
    CRCExtractor, SubsamplingExtractor, Condenser,
)

EMPTY = np.array([], dtype=np.uint8)


class TestXOR:
    def test_name(self):
        assert XORExtractor().name == "XOR"

    def test_basic(self):
        data = np.array([1, 0, 1, 1, 0, 0], dtype=np.uint8)
        assert list(XORExtractor().extract(data)) == [1, 0, 0]

    def test_empty(self):
        assert len(XORExtractor().extract(EMPTY)) == 0

    def test_single_bit(self):
        assert len(XORExtractor().extract(np.array([1], dtype=np.uint8))) == 0


class TestBlockParity:
    def test_name(self):
        assert BlockParityExtractor(block_size=4).name == "BlockParity(k=4)"

    def test_basic(self):
        data = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8)
        assert list(BlockParityExtractor(block_size=4).extract(data)) == [0, 1]

    def test_empty(self):
        assert len(BlockParityExtractor().extract(EMPTY)) == 0

    def test_shorter_than_block(self):
        assert len(BlockParityExtractor(block_size=16).extract(np.ones(4, dtype=np.uint8))) == 0


class TestBitDecimation:
    def test_name(self):
        assert BitDecimationExtractor(step=3).name == "BitDecimation(k=3)"

    def test_basic(self):
        data = np.array([1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8)
        assert list(BitDecimationExtractor(step=3).extract(data)) == [1, 1, 1]

    def test_empty(self):
        assert len(BitDecimationExtractor().extract(EMPTY)) == 0


class TestWindowedXOR:
    def test_name(self):
        assert WindowedXORExtractor(window=4).name == "WindowedXOR(w=4)"

    def test_basic(self):
        data = np.array([1, 1, 0, 1, 0, 0, 1, 0], dtype=np.uint8)
        assert list(WindowedXORExtractor(window=4).extract(data)) == [1, 1]

    def test_empty(self):
        assert len(WindowedXORExtractor().extract(EMPTY)) == 0

    def test_all_zeros(self):
        data = np.zeros(8, dtype=np.uint8)
        assert list(WindowedXORExtractor(window=4).extract(data)) == [0, 0]

    def test_all_ones(self):
        data = np.ones(8, dtype=np.uint8)
        # 4 ones XOR'd = 0
        assert list(WindowedXORExtractor(window=4).extract(data)) == [0, 0]


class TestRepetitionFilter:
    def test_name(self):
        assert RepetitionFilter().name == "RepetitionFilter"

    def test_removes_runs(self):
        data = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1], dtype=np.uint8)
        assert list(RepetitionFilter().extract(data)) == [0, 1, 0, 1]

    def test_empty(self):
        assert len(RepetitionFilter().extract(EMPTY)) == 0

    def test_single(self):
        assert list(RepetitionFilter().extract(np.array([1], dtype=np.uint8))) == [1]

    def test_no_runs(self):
        data = np.array([0, 1, 0, 1], dtype=np.uint8)
        assert list(RepetitionFilter().extract(data)) == [0, 1, 0, 1]


class TestModularReduction:
    def test_name(self):
        assert ModularReductionExtractor(block_bits=8, modulus=251).name == "ModReduction(k=8,m=251)"

    def test_output_only_bits(self):
        data = np.random.default_rng(0).integers(0, 2, size=800, dtype=np.uint8)
        result = ModularReductionExtractor(block_bits=8, modulus=127).extract(data)
        assert set(np.unique(result)).issubset({0, 1})

    def test_output_shorter_with_small_modulus(self):
        data = np.random.default_rng(0).integers(0, 2, size=800, dtype=np.uint8)
        result = ModularReductionExtractor(block_bits=8, modulus=127).extract(data)
        assert len(result) < len(data)

    def test_empty(self):
        assert len(ModularReductionExtractor().extract(EMPTY)) == 0


class TestCRC:
    def test_name(self):
        assert CRCExtractor().name == "CRC32"

    def test_output_length(self):
        data = np.random.default_rng(0).integers(0, 2, size=1000, dtype=np.uint8)
        assert len(CRCExtractor().extract(data)) == 32

    def test_deterministic(self):
        data = np.random.default_rng(0).integers(0, 2, size=1000, dtype=np.uint8)
        assert np.array_equal(CRCExtractor().extract(data), CRCExtractor().extract(data))


class TestSubsampling:
    def test_name(self):
        assert SubsamplingExtractor(output_bits=50, seed=0).name == "Subsampling(m=50)"

    def test_output_length(self):
        data = np.random.default_rng(0).integers(0, 2, size=1000, dtype=np.uint8)
        assert len(SubsamplingExtractor(output_bits=50, seed=0).extract(data)) == 50

    def test_deterministic(self):
        data = np.random.default_rng(0).integers(0, 2, size=1000, dtype=np.uint8)
        r1 = SubsamplingExtractor(output_bits=50, seed=42).extract(data)
        r2 = SubsamplingExtractor(output_bits=50, seed=42).extract(data)
        assert np.array_equal(r1, r2)

    def test_capped_at_input_length(self):
        data = np.array([1, 0, 1], dtype=np.uint8)
        assert len(SubsamplingExtractor(output_bits=100, seed=0).extract(data)) == 3


class TestCondenser:
    def test_name(self):
        assert Condenser().name == "Condenser(r=2)"

    def test_output_length(self):
        data = np.random.default_rng(0).integers(0, 2, size=1000, dtype=np.uint8)
        assert len(Condenser(rate=4).extract(data)) == 250

    def test_only_bits(self):
        data = np.random.default_rng(0).integers(0, 2, size=1000, dtype=np.uint8)
        assert set(np.unique(Condenser().extract(data))).issubset({0, 1})

    def test_matches_windowed_xor(self):
        data = np.random.default_rng(0).integers(0, 2, size=100, dtype=np.uint8)
        c = Condenser(rate=4).extract(data)
        w = WindowedXORExtractor(window=4).extract(data)
        assert np.array_equal(c, w)
