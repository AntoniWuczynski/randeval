from __future__ import annotations
import numpy as np
import pytest
from randeval.extractors.von_neumann import VonNeumannExtractor


class TestVonNeumann:
    def setup_method(self):
        self.ext = VonNeumannExtractor()

    def test_name(self):
        assert self.ext.name == "VonNeumann"

    def test_basic_pairs(self):
        data = np.array([0, 1, 1, 0, 0, 0, 1, 1], dtype=np.uint8)
        assert list(self.ext.extract(data)) == [0, 1]

    def test_all_same_returns_empty(self):
        assert len(self.ext.extract(np.zeros(100, dtype=np.uint8))) == 0

    def test_all_ones_returns_empty(self):
        assert len(self.ext.extract(np.ones(100, dtype=np.uint8))) == 0

    def test_odd_length_ignores_last(self):
        data = np.array([0, 1, 1], dtype=np.uint8)
        assert list(self.ext.extract(data)) == [0]

    def test_output_shorter_than_input(self):
        rng = np.random.default_rng(42)
        data = rng.integers(0, 2, size=10000, dtype=np.uint8)
        result = self.ext.extract(data)
        assert 0 < len(result) < len(data)

    def test_output_only_bits(self):
        rng = np.random.default_rng(42)
        result = self.ext.extract(rng.integers(0, 2, size=1000, dtype=np.uint8))
        assert set(np.unique(result)).issubset({0, 1})

    def test_debiasing_effect(self):
        rng = np.random.default_rng(99)
        data = (rng.random(10000) < 0.7).astype(np.uint8)
        result = self.ext.extract(data)
        assert abs(result.mean() - 0.5) < abs(data.mean() - 0.5)

    def test_empty_input(self):
        assert len(self.ext.extract(np.array([], dtype=np.uint8))) == 0

    def test_single_bit(self):
        assert len(self.ext.extract(np.array([1], dtype=np.uint8))) == 0
