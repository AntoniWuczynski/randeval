"""Cryptographically secure pseudorandom number generators."""

from __future__ import annotations

import hashlib
import hmac
import os

import numpy as np
from numpy.typing import NDArray

from .base import Generator
from ._utils import bytes_to_bits, require_package

SUPPORTED_HASHES: frozenset[str] = frozenset(
    {"sha256", "sha384", "sha512", "sha3_256", "sha3_512"}
)


class SystemRandom(Generator):
    """OS-level CSPRNG via `os.urandom` / `secrets`.

    Draws from the kernel entropy pool (/dev/urandom on Linux,
    CryptGenRandom on Windows). Always available on CPython.
    """

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'SystemRandom'.
        """
        return "SystemRandom"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits from the OS entropy pool via os.urandom.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        nbytes = (n + 7) // 8
        raw = os.urandom(nbytes)
        return bytes_to_bits(raw, n)

    def reset(self) -> None:
        """No-op -- OS entropy has no resettable state.

        Returns:
            None
        """
        pass


class ChaCha20(Generator):
    """ChaCha20 stream cipher used as a CSPRNG.

    256-bit key, 96-bit nonce. RFC 8439. Used by Linux's /dev/urandom
    since kernel 4.8 and by Rust's `rand` crate.
    Requires: `cryptography` package.
    """

    def __init__(self, *, key: bytes | None = None, nonce: bytes | None = None) -> None:
        """Set up ChaCha20 with a key and nonce.

        Args:
            key: 32-byte encryption key. Random if not given.
            nonce: 16-byte nonce. Random if not given.

        Raises:
            ValueError: If key is not 32 bytes or nonce is not 16 bytes.
        """
        require_package("cryptography")
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

        if key is not None and len(key) != 32:
            raise ValueError(
                f"ChaCha20 key must be exactly 32 bytes, got {len(key)}"
            )
        if nonce is not None and len(nonce) != 16:
            raise ValueError(
                f"ChaCha20 nonce must be exactly 16 bytes, got {len(nonce)}"
            )
        self._key = key if key is not None else os.urandom(32)
        self._nonce = nonce if nonce is not None else os.urandom(16)
        self._cipher = Cipher(algorithms.ChaCha20(self._key, self._nonce), mode=None)
        self._encryptor = self._cipher.encryptor()

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'ChaCha20'.
        """
        return "ChaCha20"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits by encrypting zeros to extract the keystream.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        nbytes = (n + 7) // 8
        keystream = self._encryptor.update(b"\x00" * nbytes)
        return bytes_to_bits(keystream, n)

    def reset(self) -> None:
        """Reset the encryptor to the start of the keystream.

        Returns:
            None
        """
        self._encryptor = self._cipher.encryptor()


class AESCTR_DRBG(Generator):
    """AES-256 in CTR mode as a DRBG (NIST SP 800-90A).

    The standard DRBG for FIPS 140-2 compliant systems.
    Requires: `cryptography` package.
    """

    def __init__(self, *, seed: bytes | None = None) -> None:
        """Set up AES-CTR-DRBG with a 48-byte seed.

        Args:
            seed: 48 bytes (32-byte key + 16-byte IV). Random if not given.

        Raises:
            ValueError: If seed is not exactly 48 bytes.
        """
        require_package("cryptography")
        if seed is not None and len(seed) != 48:
            raise ValueError(
                f"AES-CTR-DRBG seed must be exactly 48 bytes "
                f"(32-byte key + 16-byte IV), got {len(seed)}"
            )
        self._seed = seed if seed is not None else os.urandom(48)
        self._setup_cipher()

    def _setup_cipher(self) -> None:
        """Create the AES-CTR cipher and encryptor from the stored seed.

        Returns:
            None
        """
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        key = self._seed[:32]
        iv = self._seed[32:]
        cipher = Cipher(algorithms.AES(key), modes.CTR(iv))
        self._enc = cipher.encryptor()
        self._bytes_generated = 0

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'AES-CTR-DRBG'.
        """
        return "AES-CTR-DRBG"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits from AES-CTR keystream.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        nbytes = (n + 7) // 8
        keystream = self._enc.update(b"\x00" * nbytes)
        self._bytes_generated += nbytes
        return bytes_to_bits(keystream, n)

    def reset(self) -> None:
        """Recreate the cipher from scratch, resetting the counter.

        Returns:
            None
        """
        self._setup_cipher()


class HMAC_DRBG(Generator):
    """HMAC-based Deterministic Random Bit Generator (NIST SP 800-90A).

    Uses HMAC-SHA256 by default. Widely used in TLS and code signing.
    Uses stdlib `hmac` + `hashlib` — no extra dependencies.
    """

    def __init__(self, *, seed: bytes | None = None, hash_name: str = "sha256") -> None:
        """Set up HMAC-DRBG with optional seed and hash function.

        Args:
            seed: Entropy input bytes. Random if not given.
            hash_name: Hash algorithm name (e.g. 'sha256', 'sha3_512').

        Raises:
            ValueError: If hash_name is not in SUPPORTED_HASHES.
        """
        if hash_name not in SUPPORTED_HASHES:
            raise ValueError(
                f"Unsupported hash '{hash_name}'. "
                f"Choose from: {sorted(SUPPORTED_HASHES)}"
            )
        self._hash_name = hash_name
        self._outlen = hashlib.new(hash_name).digest_size
        self._seed = seed if seed is not None else os.urandom(self._outlen)
        self._init_state()

    def _init_state(self) -> None:
        """Initialise K and V per SP 800-90A and seed the DRBG.

        Returns:
            None
        """
        outlen = self._outlen
        self._k = b"\x00" * outlen
        self._v = b"\x01" * outlen
        self._update(self._seed)

    def _hmac(self, key: bytes, data: bytes) -> bytes:
        """Compute HMAC with the configured hash function.

        Args:
            key: HMAC key bytes.
            data: Message to authenticate.

        Returns:
            bytes: HMAC digest.
        """
        return hmac.new(key, data, self._hash_name).digest()

    def _update(self, provided_data: bytes | None = None) -> None:
        """Run the HMAC-DRBG update step (SP 800-90A section 10.1.2.2).

        Args:
            provided_data: Optional additional input to mix in.

        Returns:
            None
        """
        self._k = self._hmac(self._k, self._v + b"\x00" + (provided_data or b""))
        self._v = self._hmac(self._k, self._v)
        if provided_data:
            self._k = self._hmac(self._k, self._v + b"\x01" + provided_data)
            self._v = self._hmac(self._k, self._v)

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Name like 'HMAC-DRBG(sha256)'.
        """
        return f"HMAC-DRBG({self._hash_name})"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits by iterating the HMAC-DRBG generate loop.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        nbytes = (n + 7) // 8
        parts: list[bytes] = []
        total = 0
        while total < nbytes:
            self._v = self._hmac(self._k, self._v)
            parts.append(self._v)
            total += len(self._v)
        raw = b"".join(parts)[:nbytes]
        self._update(b"")
        return bytes_to_bits(raw, n)

    def reset(self) -> None:
        """Re-derive K and V from the original seed.

        Returns:
            None
        """
        self._init_state()


class Hash_DRBG(Generator):
    """Hash-based DRBG (NIST SP 800-90A).

    Simpler than HMAC-DRBG. Uses SHA-256 by default.
    Uses stdlib `hashlib` — no extra dependencies.
    """

    def __init__(self, *, seed: bytes | None = None, hash_name: str = "sha256") -> None:
        """Set up Hash-DRBG with optional seed and hash function.

        Args:
            seed: Entropy input bytes. Random if not given.
            hash_name: Hash algorithm name (e.g. 'sha256', 'sha3_512').

        Raises:
            ValueError: If hash_name is not in SUPPORTED_HASHES.
        """
        if hash_name not in SUPPORTED_HASHES:
            raise ValueError(
                f"Unsupported hash '{hash_name}'. "
                f"Choose from: {sorted(SUPPORTED_HASHES)}"
            )
        self._hash_name = hash_name
        digest_size = hashlib.new(hash_name).digest_size
        self._seedlen = digest_size * 8
        self._seed = seed if seed is not None else os.urandom(digest_size)
        self._init_state()

    def _hash(self, data: bytes) -> bytes:
        """Hash data with the configured hash function.

        Args:
            data: Bytes to hash.

        Returns:
            bytes: Hash digest.
        """
        return hashlib.new(self._hash_name, data).digest()

    def _init_state(self) -> None:
        """Derive V, C, and reset the reseed counter from the seed.

        Returns:
            None
        """
        seed_material = self._hash(self._seed)
        self._v = int.from_bytes(seed_material, "big")
        self._c = int.from_bytes(self._hash(b"\x00" + seed_material), "big")
        self._reseed_ctr = 1

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Name like 'Hash-DRBG(sha256)'.
        """
        return f"Hash-DRBG({self._hash_name})"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits using the Hash-DRBG hashgen + update cycle.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        nbytes = (n + 7) // 8
        parts: list[bytes] = []
        total = 0
        seedlen_bytes = (self._seedlen + 7) // 8
        data = self._v
        while total < nbytes:
            d_bytes = data.to_bytes(seedlen_bytes, "big")
            chunk = self._hash(d_bytes)
            parts.append(chunk)
            total += len(chunk)
            data = (data + 1) % (1 << self._seedlen)

        collected = b"".join(parts)[:nbytes]
        bits = bytes_to_bits(collected, n)

        v_bytes = self._v.to_bytes(seedlen_bytes, "big")
        h = int.from_bytes(self._hash(b"\x03" + v_bytes), "big")
        mod = 1 << self._seedlen
        self._v = (self._v + h + self._reseed_ctr) % mod
        self._reseed_ctr += 1

        return bits

    def reset(self) -> None:
        """Re-derive V, C, and reseed counter from the original seed.

        Returns:
            None
        """
        self._init_state()


class BlumBlumShub(Generator):
    """Blum Blum Shub — provably secure CSPRNG (under QR assumption).

    x_{n+1} = x_n^2 mod M, where M = p*q (product of two large primes
    congruent to 3 mod 4). Extracts the least significant bit.
    Extremely slow — theoretical interest primarily.
    """

    def __init__(self, *, p: int, q: int, seed: int) -> None:
        """Set up BBS with two Blum primes and a seed.

        Args:
            p: First prime, must be congruent to 3 mod 4.
            q: Second prime, must be congruent to 3 mod 4.
            seed: Initial state, must be coprime to p*q and in (1, p*q).

        Raises:
            ValueError: If primes or seed don't meet the BBS requirements.
        """
        if p < 3 or q < 3:
            raise ValueError(
                f"p and q must be primes >= 3, got p={p}, q={q}"
            )
        if p % 4 != 3:
            raise ValueError(
                f"p must be congruent to 3 mod 4, got p={p} (p%4={p%4})"
            )
        if q % 4 != 3:
            raise ValueError(
                f"q must be congruent to 3 mod 4, got q={q} (q%4={q%4})"
            )
        if p == q:
            raise ValueError("p and q must be distinct primes")
        m = p * q
        if seed <= 1 or seed >= m:
            raise ValueError(
                f"seed must be in (1, p*q={m}), got {seed}"
            )
        from math import gcd
        if gcd(seed, m) != 1:
            raise ValueError(
                f"seed must be coprime to M=p*q={m}, but gcd({seed}, {m})={gcd(seed, m)}"
            )
        self._p = p
        self._q = q
        self._m = m
        self._seed = seed
        self._x = seed

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Name like 'BlumBlumShub(20bit)'.
        """
        return f"BlumBlumShub({self._m.bit_length()}bit)"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits by squaring state mod M and extracting the LSB.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        bits = np.empty(n, dtype=np.uint8)
        x = self._x
        m = self._m
        for i in range(n):
            x = pow(x, 2, m)
            bits[i] = x & 1
        self._x = x
        return bits

    def reset(self) -> None:
        """Reset x back to the original seed.

        Returns:
            None
        """
        self._x = self._seed
