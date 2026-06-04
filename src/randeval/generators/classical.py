"""Classical (non-cryptographic) pseudorandom number generators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import random
from .base import Generator
from ._utils import uint32_to_bits, unpack_uint64

_mask64 = 0xFFFFFFFFFFFFFFFF


class LCG(Generator):
    """Linear Congruential Generator.

    Uses the recurrence: x_{n+1} = (a * x_n + c) mod m
    Extracts a single bit per iteration from `bit_index`.
    """

    def __init__(
        self,
        *,
        seed: int = 1,
        a: int = 1103515245,
        c: int = 12345,
        m: int = 2**31,
        bit_index: int = 16,
    ) -> None:
        """Create an LCG with the given parameters.

        Args:
            seed: Initial state value.
            a: Multiplier constant.
            c: Increment constant.
            m: Modulus.
            bit_index: Which bit to extract per step (0=LSB).
        """
        self._seed = seed
        self._a = a
        self._c = c
        self._m = m
        self._bit_index = bit_index
        self._x = seed

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Name string like 'LCG(a=1103515245, bit=16)'.
        """
        return f"LCG(a={self._a}, bit={self._bit_index})"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits by stepping the LCG and extracting bit_index from each state.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        bits = np.empty(n, dtype=np.uint8)
        x = self._x
        for i in range(n):
            x = (self._a * x + self._c) % self._m
            bits[i] = (x >> self._bit_index) & 1
        self._x = x
        return bits

    def reset(self) -> None:
        """Reset LCG state back to the original seed.

        Returns:
            None
        """
        self._x = self._seed


class MersenneTwister(Generator):
    """Python's `random` module (Mersenne Twister, MT19937).

    Period: 2^19937 - 1. Not cryptographically secure.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        """Configure MT19937 with an optional seed.

        Args:
            seed: Initial seed value. None means random seeding.
        """
        self._seed = seed
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'MersenneTwister'.
        """
        return "MersenneTwister"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits by pulling 32-bit words from MT19937 and unpacking.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        num_words = (n + 31) // 32
        raw = np.array([self._rng.getrandbits(32) for _ in range(num_words)], dtype=np.uint32)
        return uint32_to_bits(raw, n)

    def reset(self) -> None:
        """Re-seed the MT19937 instance with the original seed.

        Returns:
            None
        """
        self._rng = random.Random(self._seed)


class _NumpyBitGenWrapper(Generator):
    """Thin wrapper around numpy BitGenerator types.

    Subclasses just set _bg_class and _gen_name.
    """

    _bg_class: type
    _gen_name: str

    def __init__(self, *, seed: int | None = None) -> None:
        """Wrap a numpy BitGenerator with an optional seed.

        Args:
            seed: Initial seed value. None means random seeding.
        """
        self._seed = seed
        self._rng = np.random.Generator(self._bg_class(seed))

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: The generator name set by the subclass.
        """
        return self._gen_name

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits via numpy's integer sampling, then unpack to bits.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        nwords = (n + 31) // 32
        raw = self._rng.integers(0, 2**32, size=nwords, dtype=np.uint32)
        return uint32_to_bits(raw, n)

    def reset(self) -> None:
        """Recreate the BitGenerator from the original seed.

        Returns:
            None
        """
        self._rng = np.random.Generator(self._bg_class(self._seed))


class PCG64(_NumpyBitGenWrapper):
    """Permuted Congruential Generator (NumPy's default).

    Period: 2^128. Better statistical properties than MT19937.
    """
    _bg_class = np.random.PCG64
    _gen_name = "PCG64"


class Philox(_NumpyBitGenWrapper):
    """Philox4x64 — counter-based PRNG using Feistel rounds.

    Used in NumPy, TensorFlow, JAX. Statistically excellent, parallelisable.
    """
    _bg_class = np.random.Philox
    _gen_name = "Philox"


class SFC64(_NumpyBitGenWrapper):
    """Small Fast Chaotic 64-bit — Chris Doty-Humphrey.

    Available in NumPy as an alternative BitGenerator. Very fast.
    """
    _bg_class = np.random.SFC64
    _gen_name = "SFC64"


class Xorshift128Plus(Generator):
    """Xorshift128+ — used in V8, WebKit, and many JS engines.

    Two 64-bit state words. Fast but fails some BigCrush tests.
    """

    def __init__(self, *, seed: tuple[int, int] = (1, 2)) -> None:
        """Set up with a two-word seed.

        Args:
            seed: Pair of 64-bit integers. Both zero is not allowed.

        Raises:
            ValueError: If both seed words are zero.
        """
        if seed[0] == 0 and seed[1] == 0:
            raise ValueError("Xorshift128+ seed must not be all zeros")
        self._seed = seed
        self._state = list(seed)

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'Xorshift128Plus'.
        """
        return "Xorshift128Plus"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits from the xorshift128+ state, 64 bits at a time.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        bits = np.empty(n, dtype=np.uint8)
        s = self._state
        pos = 0
        while pos < n:
            s1 = s[0]
            s0 = s[1]
            s[0] = s0
            s1 ^= (s1 << 23) & _mask64
            s[1] = s1 ^ s0 ^ ((s1 >> 18) & _mask64) ^ ((s0 >> 5) & _mask64)
            result = (s[1] + s0) & _mask64
            chunk = unpack_uint64(result, n - pos)
            bits[pos:pos + len(chunk)] = chunk
            pos += len(chunk)
        return bits

    def reset(self) -> None:
        """Restore the two-word state to the original seed.

        Returns:
            None
        """
        self._state = list(self._seed)


class Xoshiro256StarStar(Generator):
    """Xoshiro256** — Blackman & Vigna's recommended general-purpose PRNG.

    256-bit state, period 2^256 - 1. Passes BigCrush.
    """

    def __init__(self, *, seed: tuple[int, int, int, int] = (1, 2, 3, 4)) -> None:
        """Set up with a four-word seed.

        Args:
            seed: Four 64-bit integers. All zeros is not allowed.

        Raises:
            ValueError: If all seed words are zero.
        """
        if all(s == 0 for s in seed):
            raise ValueError("Xoshiro256** seed must not be all zeros")
        self._seed = seed
        self._s = list(seed)

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'Xoshiro256**'.
        """
        return "Xoshiro256**"

    def _rotl(self, x: int, k: int) -> int:
        """Rotate 64-bit integer x left by k bits.

        Args:
            x: Value to rotate.
            k: Number of bit positions to shift left.

        Returns:
            int: Rotated 64-bit value.
        """
        return ((x << k) | (x >> (64 - k))) & _mask64

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits from the xoshiro256** state, 64 bits at a time.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        bits = np.empty(n, dtype=np.uint8)
        s = self._s
        pos = 0
        while pos < n:
            result = self._rotl((s[1] * 5) & _mask64, 7)
            result = (result * 9) & _mask64

            t = (s[1] << 17) & _mask64
            s[2] ^= s[0]
            s[3] ^= s[1]
            s[1] ^= s[2]
            s[0] ^= s[3]
            s[2] ^= t
            s[3] = self._rotl(s[3], 45)

            chunk = unpack_uint64(result, n - pos)
            bits[pos:pos + len(chunk)] = chunk
            pos += len(chunk)
        return bits

    def reset(self) -> None:
        """Restore the four-word state to the original seed.

        Returns:
            None
        """
        self._s = list(self._seed)


class LFSR(Generator):
    """Linear Feedback Shift Register.

    Galois or Fibonacci configuration. Taps define the feedback polynomial.
    """

    def __init__(
        self,
        *,
        seed: int = 1,
        nbits: int = 32,
        taps: tuple[int, ...] = (32, 22, 2, 1),
    ) -> None:
        """Configure LFSR with register width and feedback taps.

        Args:
            seed: Initial register value (must be non-zero).
            nbits: Width of the shift register in bits.
            taps: Feedback polynomial tap positions.

        Raises:
            ValueError: If seed is zero.
        """
        if seed == 0:
            raise ValueError("LFSR seed must be non-zero")
        self._seed = seed
        self._nbits = nbits
        self._taps = taps
        # build Galois tap mask
        self._tap_mask = 0
        for t in taps:
            if t != nbits:
                self._tap_mask |= 1 << (t - 1)
        self._reg = seed

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Name like 'LFSR(32-bit)'.
        """
        return f"LFSR({self._nbits}-bit)"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits by clocking the shift register (Galois config).

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        bits = np.empty(n, dtype=np.uint8)
        reg = self._reg
        mask = self._tap_mask
        for i in range(n):
            lsb = reg & 1
            bits[i] = lsb
            reg >>= 1
            if lsb:
                reg ^= mask
        self._reg = reg
        return bits

    def reset(self) -> None:
        """Reset the shift register to the original seed.

        Returns:
            None
        """
        self._reg = self._seed


class MiddleSquare(Generator):
    """Von Neumann's middle-square method (1946).

    Historical interest only — short period, converges to zero.
    Extracts the middle `digit_count` digits of the squared state.
    """

    def __init__(self, *, seed: int = 6239, digit_count: int = 4) -> None:
        """Configure with initial value and digit extraction width.

        Args:
            seed: Initial state value (must fit in digit_count digits).
            digit_count: Number of middle digits to extract each step.

        Raises:
            ValueError: If seed has more digits than digit_count.
        """
        if len(str(seed)) > digit_count:
            raise ValueError(f"seed {seed} has more than {digit_count} digits")
        self._seed = seed
        self._digit_count = digit_count
        self._x = seed

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Name like 'MiddleSquare(4d)'.
        """
        return f"MiddleSquare({self._digit_count}d)"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits by squaring state and extracting middle digits each step.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        bits = np.empty(n, dtype=np.uint8)
        x = self._x
        dc = self._digit_count
        for i in range(n):
            sq = x * x
            sq_str = str(sq).zfill(dc * 2)
            start = (len(sq_str) - dc) // 2
            x = int(sq_str[start:start + dc])
            bits[i] = x & 1
        self._x = x
        return bits

    def reset(self) -> None:
        """Reset state to the original seed value.

        Returns:
            None
        """
        self._x = self._seed


class MiddleSquareWeyl(Generator):
    """Middle Square Weyl Sequence — Widynski (2020).

    Fixes the original middle-square's convergence problem by adding
    a Weyl sequence counter. Passes BigCrush.
    """

    def __init__(self, *, seed: int = 0, weyl_constant: int = 0xB5AD4ECEDA1CE2A9) -> None:
        """Configure with seed and Weyl sequence constant.

        Args:
            seed: Initial state value.
            weyl_constant: Odd, irrational-derived constant for the Weyl sequence.
        """
        self._seed = seed
        self._weyl_constant = weyl_constant
        self._x = seed
        self._weyl = 0

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'MiddleSquareWeyl'.
        """
        return "MiddleSquareWeyl"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits using middle-square with Weyl sequence correction.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        bits = np.empty(n, dtype=np.uint8)
        x = self._x
        w = self._weyl
        wc = self._weyl_constant
        pos = 0
        while pos < n:
            w = (w + wc) & _mask64
            x = (x * x + w) & _mask64
            x = ((x >> 32) | (x << 32)) & _mask64
            chunk = unpack_uint64(x, n - pos)
            bits[pos:pos + len(chunk)] = chunk
            pos += len(chunk)
        self._x = x
        self._weyl = w
        return bits

    def reset(self) -> None:
        """Reset state and Weyl counter to initial values.

        Returns:
            None
        """
        self._x = self._seed
        self._weyl = 0


class WichmannHill(Generator):
    """Wichmann-Hill (1982) — combination of three LCGs.

    Historical interest. Short period (~7 x 10^12). Was Python's default
    before MT19937.
    """

    def __init__(
        self,
        *,
        seed: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        """Set up three LCG sub-states from the seed triple.

        Args:
            seed: Triple of initial values for the three LCG sub-generators.
        """
        self._seed = seed
        self._s1, self._s2, self._s3 = seed

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'WichmannHill'.
        """
        return "WichmannHill"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits by combining three LCG outputs and thresholding at 0.5.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        bits = np.empty(n, dtype=np.uint8)
        s1, s2, s3 = self._s1, self._s2, self._s3
        for i in range(n):
            s1 = (171 * s1) % 30269
            s2 = (172 * s2) % 30307
            s3 = (170 * s3) % 30323
            r = (s1 / 30269 + s2 / 30307 + s3 / 30323) % 1.0
            bits[i] = int(r >= 0.5)
        self._s1, self._s2, self._s3 = s1, s2, s3
        return bits

    def reset(self) -> None:
        """Restore all three sub-states to the original seed.

        Returns:
            None
        """
        self._s1, self._s2, self._s3 = self._seed
