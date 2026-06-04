from __future__ import annotations
import numpy as np
import pytest
from randeval.extractors._utils import (
    gf2_matmul,
    random_binary_seed,
    next_prime_with_primitive_root_2,
    estimate_bias,
    bits_to_int,
    int_to_bits,
)


class TestGF2Matmul:
    def test_identity(self):
        mat = np.eye(3, dtype=np.uint8)
        vec = np.array([1, 0, 1], dtype=np.uint8)
        assert np.array_equal(gf2_matmul(mat, vec), vec)

    def test_known_product(self):
        mat = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        vec = np.array([1, 1, 0], dtype=np.uint8)
        expected = np.array([0, 1], dtype=np.uint8)
        assert np.array_equal(gf2_matmul(mat, vec), expected)

    def test_output_is_mod2(self):
        mat = np.ones((2, 4), dtype=np.uint8)
        vec = np.ones(4, dtype=np.uint8)
        result = gf2_matmul(mat, vec)
        assert all(b in (0, 1) for b in result)


class TestRandomBinarySeed:
    def test_length(self):
        assert len(random_binary_seed(100)) == 100

    def test_dtype(self):
        assert random_binary_seed(50).dtype == np.uint8

    def test_only_bits(self):
        assert set(np.unique(random_binary_seed(1000))).issubset({0, 1})

    def test_deterministic_with_rng(self):
        s1 = random_binary_seed(100, np.random.default_rng(42))
        s2 = random_binary_seed(100, np.random.default_rng(42))
        assert np.array_equal(s1, s2)


class TestNextPrimeWithPrimitiveRoot2:
    def test_small(self):
        assert next_prime_with_primitive_root_2(3) == 3

    def test_known_value(self):
        assert next_prime_with_primitive_root_2(5) == 5

    def test_skip_non_qualifying(self):
        p = next_prime_with_primitive_root_2(4)
        assert p >= 4


class TestEstimateBias:
    def test_fair(self):
        bits = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint8)
        assert estimate_bias(bits) == pytest.approx(0.5)

    def test_all_ones(self):
        assert estimate_bias(np.ones(100, dtype=np.uint8)) == pytest.approx(1.0)

    def test_all_zeros(self):
        assert estimate_bias(np.zeros(100, dtype=np.uint8)) == pytest.approx(0.0)


class TestBitsIntConversion:
    def test_roundtrip(self):
        for val in [0, 1, 7, 255, 1023, 65535, 2**20 - 1]:
            width = max(val.bit_length(), 1)
            assert bits_to_int(int_to_bits(val, width)) == val

    def test_known(self):
        bits = int_to_bits(5, 3)
        assert list(bits) == [1, 0, 1]
        assert bits_to_int(bits) == 5

    def test_zero_width_one(self):
        bits = int_to_bits(0, 1)
        assert list(bits) == [0]
        assert bits_to_int(bits) == 0

    def test_large_value(self):
        val = 2**64 - 1
        bits = int_to_bits(val, 64)
        assert bits_to_int(bits) == val
