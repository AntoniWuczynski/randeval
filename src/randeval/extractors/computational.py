"""Computational extractors: HMAC-based and SHA conditioning."""
from __future__ import annotations

import hashlib
import hmac
import os

import numpy as np
from numpy.typing import NDArray

from .base import Extractor


class HMACExtractor(Extractor):
    """HMAC-based computational extractor (keyed).

    NOTE: Always outputs exactly hash_len bits (256 for SHA-256) regardless of
    input size. At 256 bits, most statistical tests lack power to discriminate —
    expect ~50% pass rate. For longer output, use SHA-Conditioner instead.
    """

    def __init__(self, *, key: bytes | None = None, hash_name: str = "sha256") -> None:
        self._key = key if key is not None else os.urandom(32)
        self._hash_name = hash_name
        self._hash_len = hashlib.new(hash_name).digest_size * 8

    @property
    def name(self) -> str:
        return f"HMAC({self._hash_name})"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        packed = np.packbits(data).tobytes()
        digest = hmac.new(self._key, packed, self._hash_name).digest()
        return np.unpackbits(np.frombuffer(digest, dtype=np.uint8))


class SHAConditioner(Extractor):
    """Hash-based conditioner per NIST SP 800-90B section 3.1.5."""

    def __init__(self, hash_name: str = "sha256") -> None:
        self._hash_name = hash_name
        self._digest_size = hashlib.new(hash_name).digest_size
        self._hash_bits = self._digest_size * 8

    @property
    def name(self) -> str:
        return f"{self._hash_name.upper()}-Conditioner"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        packed = np.packbits(data).tobytes()
        n_blocks = max(1, len(data) // (2 * self._hash_bits))
        parts: list[bytes] = []
        for i in range(n_blocks):
            h = hashlib.new(self._hash_name)
            h.update(i.to_bytes(4, "big"))
            h.update(packed)
            parts.append(h.digest())
        raw = b"".join(parts)
        return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
