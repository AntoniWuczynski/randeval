from __future__ import annotations
import numpy as np
import pytest
from randeval.extractors.adaptive import (
    MinEntropyExtractor, FuzzyExtractor, ArithmeticCodingExtractor,
)


class TestMinEntropyExtractor:
    def test_name(self):
        assert MinEntropyExtractor().name == "MinEntropyExtractor"

    def test_output_shorter(self):
        data = np.random.default_rng(42).integers(0, 2, size=10000, dtype=np.uint8)
        result = MinEntropyExtractor(seed=0).extract(data)
        assert 0 < len(result) < len(data)

    def test_only_bits(self):
        data = np.random.default_rng(42).integers(0, 2, size=1000, dtype=np.uint8)
        assert set(np.unique(MinEntropyExtractor(seed=0).extract(data))).issubset({0, 1})

    def test_biased_produces_less(self):
        fair = np.random.default_rng(42).integers(0, 2, size=10000, dtype=np.uint8)
        biased = (np.random.default_rng(43).random(10000) < 0.9).astype(np.uint8)
        fair_out = len(MinEntropyExtractor(seed=0).extract(fair))
        biased_out = len(MinEntropyExtractor(seed=0).extract(biased))
        assert biased_out < fair_out

    def test_all_same_returns_empty(self):
        assert len(MinEntropyExtractor().extract(np.zeros(100, dtype=np.uint8))) == 0

    def test_short_input(self):
        assert len(MinEntropyExtractor().extract(np.array([1, 0], dtype=np.uint8))) == 0


class TestFuzzyExtractor:
    def test_name(self):
        assert FuzzyExtractor(seed=0).name == "FuzzyExtractor"

    def test_gen_produces_key_and_helper(self):
        data = np.random.default_rng(42).integers(0, 2, size=1000, dtype=np.uint8)
        key, helper = FuzzyExtractor(seed=0).generate(data)
        assert len(key) > 0
        assert len(helper) == len(data)

    def test_rep_recovers_key_exact(self):
        # exact reproduction (no noise) should recover the key
        rng = np.random.default_rng(42)
        data = rng.integers(0, 2, size=1000, dtype=np.uint8)
        ext = FuzzyExtractor(seed=0)
        key, helper = ext.generate(data)
        recovered = ext.reproduce(data, helper)
        assert np.array_equal(key, recovered)

    def test_extract_returns_key(self):
        data = np.random.default_rng(42).integers(0, 2, size=1000, dtype=np.uint8)
        assert len(FuzzyExtractor(seed=0).extract(data)) > 0

    def test_too_noisy_fails(self):
        rng = np.random.default_rng(42)
        data = rng.integers(0, 2, size=1000, dtype=np.uint8)
        ext = FuzzyExtractor(seed=0)
        key, helper = ext.generate(data)
        noisy = data.copy()
        flip_idx = rng.choice(len(data), size=len(data) // 2, replace=False)
        noisy[flip_idx] ^= 1
        recovered = ext.reproduce(noisy, helper)
        assert not np.array_equal(key, recovered)


class TestArithmeticCoding:
    def test_name(self):
        assert ArithmeticCodingExtractor(bias=0.7).name == "ArithmeticCoding(p=0.70)"

    def test_output_only_bits(self):
        data = (np.random.default_rng(42).random(1000) < 0.7).astype(np.uint8)
        result = ArithmeticCodingExtractor(bias=0.7).extract(data)
        assert set(np.unique(result)).issubset({0, 1})

    def test_near_optimal_efficiency(self):
        import math
        p = 0.7
        h = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        data = (np.random.default_rng(42).random(10000) < p).astype(np.uint8)
        result = ArithmeticCodingExtractor(bias=p).extract(data)
        assert len(result) > h * len(data) * 0.5

    def test_invalid_bias(self):
        with pytest.raises(ValueError):
            ArithmeticCodingExtractor(bias=0.0)
        with pytest.raises(ValueError):
            ArithmeticCodingExtractor(bias=1.0)

    def test_empty(self):
        assert len(ArithmeticCodingExtractor(bias=0.5).extract(np.array([], dtype=np.uint8))) == 0
