from __future__ import annotations
import numpy as np
import pytest

try:
    import cryptomite
    HAS_CRYPTOMITE = True
except ImportError:
    HAS_CRYPTOMITE = False

skip_no_cryptomite = pytest.mark.skipif(not HAS_CRYPTOMITE, reason="cryptomite not installed")

from randeval.extractors.cryptomite_ext import (
    CryptoMiteToeplitz, CryptoMiteCirculant, CryptoMiteDodis, CryptoMiteTrevisan,
)

RNG_DATA = np.random.default_rng(42).integers(0, 2, size=1000, dtype=np.uint8)


@skip_no_cryptomite
class TestCryptoMiteToeplitz:
    def test_name(self):
        assert CryptoMiteToeplitz(min_entropy=800).name == "CryptoMite-Toeplitz"

    def test_extraction(self):
        result = CryptoMiteToeplitz(min_entropy=800).extract(RNG_DATA)
        assert len(result) > 0
        assert set(np.unique(result)).issubset({0, 1})


@skip_no_cryptomite
class TestCryptoMiteCirculant:
    def test_name(self):
        assert CryptoMiteCirculant(min_entropy=800).name == "CryptoMite-Circulant"

    def test_extraction(self):
        result = CryptoMiteCirculant(min_entropy=800).extract(RNG_DATA)
        assert len(result) > 0
        assert set(np.unique(result)).issubset({0, 1})


@skip_no_cryptomite
class TestCryptoMiteDodis:
    def test_name(self):
        assert CryptoMiteDodis(min_entropy=800).name == "CryptoMite-Dodis"

    def test_extraction(self):
        result = CryptoMiteDodis(min_entropy=800).extract(RNG_DATA)
        assert len(result) > 0
        assert set(np.unique(result)).issubset({0, 1})


@skip_no_cryptomite
class TestCryptoMiteTrevisan:
    def test_name(self):
        assert CryptoMiteTrevisan(min_entropy=800).name == "CryptoMite-Trevisan"

    def test_extraction(self):
        result = CryptoMiteTrevisan(min_entropy=800, error=1e-4).extract(RNG_DATA)
        assert len(result) > 0
        assert set(np.unique(result)).issubset({0, 1})
