"""CryptoMite library wrappers — quantum-grade seeded extraction."""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from .base import Extractor
from ._utils import require_package


def _output_len(k: float, eps: float, n: int) -> int:
    """Compute extractor output length from min-entropy and error bound."""
    k1 = k if k > 1 else k * n
    m = max(1, int(k1 - 2 * math.log2(1 / eps)))
    return min(m, n - 1)


class CryptoMiteToeplitz(Extractor):
    """Toeplitz hashing via the CryptoMite library (Quantinuum)."""

    def __init__(self, min_entropy: float = 0.9, epsilon: float = 2**-32) -> None:
        self._k = min_entropy
        self._eps = epsilon

    @property
    def name(self) -> str:
        return "CryptoMite-Toeplitz"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        require_package("cryptomite", pip_extra="extraction")
        from cryptomite.toeplitz import Toeplitz

        n = len(data)
        m = _output_len(self._k, self._eps, n)
        ext = Toeplitz(n, m)
        seed_len = n + m - 1
        seed = np.random.default_rng().integers(0, 2, size=seed_len).tolist()
        result = ext.extract(data.tolist(), seed)
        return np.array(result, dtype=np.uint8)


class CryptoMiteCirculant(Extractor):
    """Circulant matrix extractor via CryptoMite."""

    def __init__(self, min_entropy: float = 0.9, epsilon: float = 2**-32) -> None:
        self._k = min_entropy
        self._eps = epsilon

    @property
    def name(self) -> str:
        return "CryptoMite-Circulant"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        require_package("cryptomite", pip_extra="extraction")
        from cryptomite.circulant import Circulant
        import cryptomite.utils as cm_utils

        n = len(data)
        n_valid = cm_utils.previous_prime(n + 1)
        trimmed = data[:n_valid].tolist()
        m = _output_len(self._k, self._eps, n_valid)
        ext = Circulant(n_valid, m)
        seed = np.random.default_rng().integers(0, 2, size=n_valid + 1).tolist()
        result = ext.extract(trimmed, seed)
        return np.array(result, dtype=np.uint8)


class CryptoMiteDodis(Extractor):
    """Dodis et al. two-source extractor via CryptoMite."""

    def __init__(self, min_entropy: float = 0.9, epsilon: float = 2**-32) -> None:
        self._k = min_entropy
        self._eps = epsilon

    @property
    def name(self) -> str:
        return "CryptoMite-Dodis"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        require_package("cryptomite", pip_extra="extraction")
        from cryptomite.dodis import Dodis
        import cryptomite.utils as cm_utils

        n = len(data)
        n_valid = cm_utils.previous_na_set(n)
        trimmed = data[:n_valid].tolist()
        m = _output_len(self._k, self._eps, n_valid)
        ext = Dodis(n_valid, m)
        seed = np.random.default_rng().integers(0, 2, size=n_valid).tolist()
        result = ext.extract(trimmed, seed)
        return np.array(result, dtype=np.uint8)


class CryptoMiteTrevisan(Extractor):
    """Trevisan's extractor via CryptoMite — shortest seed, theoretical optimum."""

    # cryptomite's Trevisan is O(m * polylog n) per output bit and gets
    # impractical past ~300k bits. Matches the 2025 research project cap.
    DEFAULT_MAX_INPUT = 100_000

    def __init__(
        self,
        min_entropy: float = 0.9,
        error: float = 1e-4,
        *,
        cap_large_inputs: bool = True,
        max_input: int = DEFAULT_MAX_INPUT,
    ) -> None:
        self._k = min_entropy
        self._error = error
        self._cap = cap_large_inputs
        self._max_input = max_input

    @property
    def name(self) -> str:
        return "CryptoMite-Trevisan"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        require_package("cryptomite", pip_extra="extraction")
        from cryptomite.trevisan import Trevisan

        if self._cap and len(data) > self._max_input:
            data = data[: self._max_input]

        n = len(data)
        k1 = self._k if self._k > 1 else self._k * n
        ext = Trevisan(n, int(k1), self._error)
        seed_len = ext.ext.get_seed_length()
        seed = np.random.default_rng().integers(0, 2, size=seed_len).tolist()
        result = ext.extract(data.tolist(), seed)
        return np.array(result, dtype=np.uint8)
