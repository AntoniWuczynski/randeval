"""Convert a numpy bit array into a stream of uniform floats in [0, 1).

Both the Mitsuba sampler and the Python fallback integrator pull bits through
this object. Wrapping is on by default — once we run out we cycle back to the
start and bump a counter so the report can flag it.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class BitFloatStream:
    """Sequential bit-to-float source backed by a flat uint8 array.

    Each next_float() draw consumes `bits_per_value` bits and returns a float
    in [0, 1). The stream wraps when exhausted; check `wraps` after rendering
    to know if the budget was tight.
    """

    def __init__(self, bits: NDArray[np.uint8], bits_per_value: int = 32) -> None:
        if bits.ndim != 1:
            raise ValueError(f"need 1-D bit array, got {bits.shape}")
        if bits_per_value < 1 or bits_per_value > 53:
            raise ValueError(f"bits_per_value must be in 1..53, got {bits_per_value}")
        self._bits = bits.astype(np.uint8, copy=False)
        self._k = bits_per_value
        self._scale = 1.0 / float(1 << bits_per_value)
        self._pos = 0
        self.wraps = 0

    def __len__(self) -> int:
        return int(self._bits.size)

    @property
    def position(self) -> int:
        return self._pos

    def reset(self) -> None:
        self._pos = 0
        self.wraps = 0

    def next_float(self) -> float:
        n = self._bits.size
        if self._pos + self._k > n:
            self._pos = 0
            self.wraps += 1
        block = self._bits[self._pos : self._pos + self._k]
        self._pos += self._k
        v = 0
        for b in block:
            v = (v << 1) | int(b)
        return v * self._scale

    def next_floats(self, count: int) -> NDArray[np.float64]:
        out = np.empty(count, dtype=np.float64)
        for i in range(count):
            out[i] = self.next_float()
        return out

    def floats_consumed(self) -> int:
        return self.wraps * (len(self) // self._k) + (self._pos // self._k)

    def fork(self) -> "BitFloatStream":
        """Return a fresh view starting at the current position.

        Mitsuba will call sampler.fork() to spawn workers; we hand them a
        stream that picks up where the parent left off rather than restarting.
        """
        sub = BitFloatStream(self._bits, self._k)
        sub._pos = self._pos
        return sub
