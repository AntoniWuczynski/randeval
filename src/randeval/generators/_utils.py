"""Shared utilities for generator implementations."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bytes_to_bits(data: bytes | NDArray[np.uint8], n: int) -> NDArray[np.uint8]:
    """Convert raw bytes to a truncated bit array of length n.

    Args:
        data: Raw bytes or uint8 numpy array to unpack.
        n: Number of bits to keep from the unpacked result.

    Returns:
        NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
    """
    if isinstance(data, (bytes, bytearray)):
        arr = np.frombuffer(data, dtype=np.uint8)
    else:
        arr = data
    return np.unpackbits(arr)[:n]


def uint32_to_bits(raw: NDArray[np.uint32], n: int) -> NDArray[np.uint8]:
    """Unpack uint32 words to individual bits, truncated to n.

    Args:
        raw: Array of uint32 values to unpack.
        n: Number of bits to keep.

    Returns:
        NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
    """
    return np.unpackbits(raw.view(np.uint8), bitorder="big")[:n]


def unpack_uint64(val: int, max_bits: int = 64) -> NDArray[np.uint8]:
    """Extract individual bits from a 64-bit integer as a numpy array.

    Args:
        val: 64-bit integer to unpack.
        max_bits: Maximum number of bits to extract (capped at 64).

    Returns:
        NDArray[np.uint8]: Array of 0s and 1s, length min(64, max_bits).
    """
    take = min(64, max_bits)
    word_bytes = (val & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little")
    return np.unpackbits(np.frombuffer(word_bytes, dtype=np.uint8), bitorder="little")[:take]


def require_package(package: str, pip_name: str | None = None) -> None:
    """Check that a package is importable, raise with install hint if not.

    Args:
        package: Python package name to try importing.
        pip_name: Pip install target if different from package name.

    Returns:
        None

    Raises:
        ImportError: If the package cannot be imported.
    """
    try:
        __import__(package)
    except ImportError as exc:
        install = pip_name or package
        raise ImportError(
            f"'{package}' is required. Install with: pip install {install}"
        ) from exc
