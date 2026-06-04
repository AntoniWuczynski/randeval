from __future__ import annotations
import numpy as np
import pytest
from randeval.extractors.computational import HMACExtractor, SHAConditioner

RNG_DATA = np.random.default_rng(0).integers(0, 2, size=1000, dtype=np.uint8)


class TestHMACExtractor:
    def setup_method(self):
        self.ext = HMACExtractor(key=b"test_key_32bytes_padded_here!!!!!")

    def test_name(self):
        assert self.ext.name == "HMAC(sha256)"

    def test_output_length(self):
        assert len(self.ext.extract(RNG_DATA)) == 256

    def test_deterministic(self):
        assert np.array_equal(self.ext.extract(RNG_DATA), self.ext.extract(RNG_DATA))

    def test_different_keys_differ(self):
        e1 = HMACExtractor(key=b"key_a_padded_to_be_long_enough!!")
        e2 = HMACExtractor(key=b"key_b_padded_to_be_long_enough!!")
        assert not np.array_equal(e1.extract(RNG_DATA), e2.extract(RNG_DATA))

    def test_only_bits(self):
        assert set(np.unique(self.ext.extract(RNG_DATA))).issubset({0, 1})

    def test_random_key(self):
        assert len(HMACExtractor().extract(RNG_DATA)) == 256


class TestSHAConditioner:
    def setup_method(self):
        self.ext = SHAConditioner()

    def test_name(self):
        assert self.ext.name == "SHA256-Conditioner"

    def test_output_length_divisible(self):
        result = self.ext.extract(RNG_DATA)
        assert len(result) > 0
        assert len(result) % 256 == 0

    def test_deterministic(self):
        assert np.array_equal(self.ext.extract(RNG_DATA), self.ext.extract(RNG_DATA))

    def test_only_bits(self):
        assert set(np.unique(self.ext.extract(RNG_DATA))).issubset({0, 1})

    def test_custom_hash(self):
        ext = SHAConditioner(hash_name="sha512")
        assert ext.name == "SHA512-Conditioner"
        assert len(ext.extract(RNG_DATA)) % 512 == 0
