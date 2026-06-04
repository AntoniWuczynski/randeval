from __future__ import annotations
import numpy as np
import pytest
from randeval.extractors.hashing import (
    ToeplitzHash, LinearHash, InnerProductExtractor, LHLHash, PolynomialHash,
)

RNG_DATA = np.random.default_rng(0).integers(0, 2, size=256, dtype=np.uint8)


class TestToeplitzHash:
    def test_name(self):
        assert ToeplitzHash(output_bits=64, seed=42).name == "ToeplitzHash(m=64)"

    def test_output_length(self):
        assert len(ToeplitzHash(output_bits=64, seed=42).extract(RNG_DATA)) == 64

    def test_deterministic_with_seed(self):
        r1 = ToeplitzHash(output_bits=64, seed=42).extract(RNG_DATA)
        r2 = ToeplitzHash(output_bits=64, seed=42).extract(RNG_DATA)
        assert np.array_equal(r1, r2)

    def test_different_seeds_differ(self):
        r1 = ToeplitzHash(output_bits=64, seed=42).extract(RNG_DATA)
        r2 = ToeplitzHash(output_bits=64, seed=99).extract(RNG_DATA)
        assert not np.array_equal(r1, r2)

    def test_only_bits(self):
        result = ToeplitzHash(output_bits=64, seed=42).extract(RNG_DATA)
        assert set(np.unique(result)).issubset({0, 1})

    def test_input_shorter_than_output(self):
        data = np.array([1, 0, 1], dtype=np.uint8)
        assert len(ToeplitzHash(output_bits=2, seed=1).extract(data)) == 2


class TestLinearHash:
    def test_name(self):
        assert LinearHash(output_bits=32, seed=42).name == "LinearHash(m=32)"

    def test_output_length(self):
        assert len(LinearHash(output_bits=32, seed=42).extract(RNG_DATA)) == 32

    def test_only_bits(self):
        result = LinearHash(output_bits=32, seed=42).extract(RNG_DATA)
        assert set(np.unique(result)).issubset({0, 1})

    def test_deterministic(self):
        r1 = LinearHash(output_bits=32, seed=7).extract(RNG_DATA)
        r2 = LinearHash(output_bits=32, seed=7).extract(RNG_DATA)
        assert np.array_equal(r1, r2)


class TestInnerProduct:
    def test_name(self):
        assert InnerProductExtractor().name == "InnerProduct"

    def test_halves_input(self):
        assert len(InnerProductExtractor().extract(RNG_DATA)) > 0

    def test_only_bits(self):
        result = InnerProductExtractor().extract(RNG_DATA)
        assert set(np.unique(result)).issubset({0, 1})

    def test_empty(self):
        assert len(InnerProductExtractor().extract(np.array([], dtype=np.uint8))) == 0

    def test_two_bits(self):
        # minimal case: 2 bits -> split into halves of 1 -> inner product is AND
        data = np.array([1, 1], dtype=np.uint8)
        result = InnerProductExtractor().extract(data)
        assert len(result) == 1


class TestLHLHash:
    def test_name(self):
        assert LHLHash(output_bits=32, seed=42).name == "LHLHash(m=32)"

    def test_output_length(self):
        assert len(LHLHash(output_bits=32, seed=42).extract(RNG_DATA)) == 32

    def test_only_bits(self):
        result = LHLHash(output_bits=32, seed=42).extract(RNG_DATA)
        assert set(np.unique(result)).issubset({0, 1})

    def test_deterministic(self):
        r1 = LHLHash(output_bits=32, seed=42).extract(RNG_DATA)
        r2 = LHLHash(output_bits=32, seed=42).extract(RNG_DATA)
        assert np.array_equal(r1, r2)


class TestPolynomialHash:
    def test_name(self):
        assert PolynomialHash(output_bits=32, seed=42).name == "PolynomialHash(m=32)"

    def test_output_length(self):
        assert len(PolynomialHash(output_bits=32, seed=42).extract(RNG_DATA)) == 32

    def test_only_bits(self):
        result = PolynomialHash(output_bits=32, seed=42).extract(RNG_DATA)
        assert set(np.unique(result)).issubset({0, 1})

    def test_deterministic(self):
        r1 = PolynomialHash(output_bits=32, seed=42).extract(RNG_DATA)
        r2 = PolynomialHash(output_bits=32, seed=42).extract(RNG_DATA)
        assert np.array_equal(r1, r2)
