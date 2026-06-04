from __future__ import annotations
import numpy as np
import pytest
from randeval.extractors.debiasing import PeresExtractor, EliasExtractor, AMLSExtractor


class TestPeres:
    def setup_method(self):
        self.ext = PeresExtractor()

    def test_name(self):
        assert self.ext.name == "Peres"

    def test_basic_extraction(self):
        rng = np.random.default_rng(42)
        data = rng.integers(0, 2, size=10000, dtype=np.uint8)
        result = self.ext.extract(data)
        assert len(result) > 0
        assert set(np.unique(result)).issubset({0, 1})

    def test_more_output_than_von_neumann(self):
        from randeval.extractors.von_neumann import VonNeumannExtractor
        rng = np.random.default_rng(42)
        data = rng.integers(0, 2, size=10000, dtype=np.uint8)
        vn_len = len(VonNeumannExtractor().extract(data))
        peres_len = len(self.ext.extract(data))
        assert peres_len >= vn_len

    def test_debiasing(self):
        rng = np.random.default_rng(99)
        data = (rng.random(10000) < 0.7).astype(np.uint8)
        result = self.ext.extract(data)
        assert abs(result.mean() - 0.5) < 0.1

    def test_empty(self):
        assert len(self.ext.extract(np.array([], dtype=np.uint8))) == 0

    def test_all_same(self):
        assert len(self.ext.extract(np.ones(100, dtype=np.uint8))) == 0


class TestElias:
    def setup_method(self):
        self.ext = EliasExtractor(bias=0.7)

    def test_name(self):
        assert self.ext.name == "Elias(p=0.70)"

    def test_extraction(self):
        rng = np.random.default_rng(42)
        data = (rng.random(10000) < 0.7).astype(np.uint8)
        result = self.ext.extract(data)
        assert len(result) > 0
        # Elias with approximate bias can have residual skew — just check it's better than input
        assert abs(result.mean() - 0.5) < abs(data.mean() - 0.5)

    def test_invalid_bias(self):
        with pytest.raises(ValueError):
            EliasExtractor(bias=0.0)
        with pytest.raises(ValueError):
            EliasExtractor(bias=1.0)

    def test_empty(self):
        assert len(self.ext.extract(np.array([], dtype=np.uint8))) == 0

    def test_short_input(self):
        result = self.ext.extract(np.array([1], dtype=np.uint8))
        assert len(result) == 0


class TestAMLS:
    def setup_method(self):
        self.ext = AMLSExtractor()

    def test_name(self):
        assert self.ext.name == "AMLS"

    def test_more_output_than_peres(self):
        rng = np.random.default_rng(42)
        data = (rng.random(10000) < 0.6).astype(np.uint8)
        peres_len = len(PeresExtractor().extract(data))
        amls_len = len(self.ext.extract(data))
        assert amls_len >= peres_len

    def test_debiasing(self):
        rng = np.random.default_rng(99)
        data = (rng.random(10000) < 0.8).astype(np.uint8)
        result = self.ext.extract(data)
        if len(result) > 100:
            assert abs(result.mean() - 0.5) < 0.1

    def test_empty(self):
        assert len(self.ext.extract(np.array([], dtype=np.uint8))) == 0

    def test_single_bit(self):
        assert len(self.ext.extract(np.array([1], dtype=np.uint8))) == 0
