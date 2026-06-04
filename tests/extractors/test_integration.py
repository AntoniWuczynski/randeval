"""Smoke test: generate -> extract -> verify for all extractors."""
from __future__ import annotations
import numpy as np
import pytest
from randeval import RandomSequence
from randeval.extractors import default_extractors, list_all


class TestExtractorIntegration:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.seq = RandomSequence(rng.integers(0, 2, size=10000, dtype=np.uint8))

    def test_all_extractors_produce_bits(self):
        for ext in default_extractors():
            result = self.seq.extract(ext)
            assert len(result) > 0, f"{ext.name} produced empty output"
            assert set(np.unique(result.data)).issubset({0, 1}), f"{ext.name} has non-bit values"

    def test_extract_all(self):
        results = self.seq.extract_all()
        assert len(results) == len(default_extractors())
        for name, seq in results.items():
            assert len(seq) > 0, f"{name} produced empty output"

    def test_extract_then_test(self):
        from randeval.tests_statistical import FrequencyTest
        ext_seq = self.seq.extract(default_extractors()[0])
        result = ext_seq.test(FrequencyTest())
        assert result.p_value is not None

    def test_metadata_preserved(self):
        from randeval.extractors import VonNeumannExtractor
        ext_seq = self.seq.extract(VonNeumannExtractor())
        assert ext_seq.metadata.extractor_name == "VonNeumann"
        assert ext_seq.metadata.original_length == 10000

    def test_chaining(self):
        from randeval.extractors import VonNeumannExtractor, XORExtractor
        chained = self.seq.extract(VonNeumannExtractor()).extract(XORExtractor())
        assert len(chained) > 0
        assert chained.metadata.extractor_name == "XOR"
