"""True Random Number Generators (TRNGs).

These derive randomness from physical phenomena — hardware devices,
environmental noise, or remote entropy services. Many require network
access or platform-specific hardware.
"""

from __future__ import annotations

import hashlib
import os
import platform
import struct
import sys
import time
import warnings
from pathlib import Path
from typing import IO

import numpy as np
from numpy.typing import NDArray

from abc import abstractmethod

from .base import Generator
from ._utils import bytes_to_bits, unpack_uint64, require_package


# ── Shared checks ─────────────────────────────────────────────

def _warn_intrusive(generator_name: str, action: str, detail: str) -> None:
    """Emit a UserWarning about a potentially intrusive side effect.

    Args:
        generator_name: Name of the generator triggering the warning.
        action: What the generator is about to do.
        detail: Extra context (rate limits, permissions, etc.).

    Returns:
        None
    """
    warnings.warn(
        f"[{generator_name}] {action}. {detail}",
        UserWarning,
        stacklevel=3,
    )


def _check_x86_64() -> None:
    """Raise if not running on an x86_64 CPU.

    Returns:
        None

    Raises:
        RuntimeError: If the platform is not x86_64/amd64.
    """
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        raise RuntimeError(
            f"This generator requires an x86_64 CPU, "
            f"but the current platform is '{machine}'"
        )


def _check_cpuid_feature(feature: str) -> None:
    """Best-effort check for a CPU feature via cpuid (x86_64 only).

    Falls back to a warning if detection is not possible.

    Args:
        feature: CPU feature to check for (e.g. 'RDRAND', 'RDSEED').

    Returns:
        None
    """
    _check_x86_64()
    try:
        import cpuid  # noqa: F401 — optional, not a hard requirement
    except ImportError:
        import warnings
        warnings.warn(
            f"Cannot verify CPU support for {feature}: 'cpuid' package not "
            f"installed. The generator will attempt to proceed but may fail "
            f"at runtime. Install with: pip install cpuid",
            RuntimeWarning,
            stacklevel=3,
        )


def _check_linux() -> None:
    """Raise if not running on Linux.

    Returns:
        None

    Raises:
        RuntimeError: If sys.platform is not 'linux'.
    """
    if sys.platform != "linux":
        raise RuntimeError(
            f"This generator requires Linux, "
            f"but the current platform is '{sys.platform}'"
        )


def _check_device_readable(path: Path) -> None:
    """Raise if a device file does not exist or is not readable.

    Args:
        path: Filesystem path to the device file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the device doesn't exist.
        PermissionError: If the device isn't readable.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Device '{path}' does not exist. Ensure the appropriate "
            f"hardware RNG kernel module is loaded (e.g. tpm-rng, virtio-rng)."
        )
    if not os.access(path, os.R_OK):
        raise PermissionError(
            f"Cannot read '{path}'. Try running with elevated privileges "
            f"or add your user to the appropriate group: "
            f"sudo usermod -aG {path.name} $USER"
        )


# ── Remote / API-based TRNGs ──────────────────────────────────

class RandomOrgGenerator(Generator):
    """Random.org — atmospheric noise.

    Uses the Random.org JSON-RPC API (requires an API key for the
    paid tier; free tier has daily quota). Bits are generated from
    radio atmospheric noise.
    Requires: network access, `requests` package.
    """

    def __init__(self, *, api_key: str | None = None) -> None:
        """Set up Random.org client.

        Args:
            api_key: API key for the paid JSON-RPC tier. None uses the free tier.
        """
        require_package("requests")
        _warn_intrusive(
            "Random.org",
            "Will make HTTPS requests to api.random.org",
            "Subject to rate limits; paid tier requires an API key.",
        )
        self._api_key = api_key

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'Random.org (Atmospheric Noise)'.
        """
        return "Random.org (Atmospheric Noise)"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Fetch n random bits from Random.org over HTTPS.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        import requests

        bits: list[int] = []
        if self._api_key:
            url = "https://api.random.org/json-rpc/4/invoke"
            remaining = n
            while remaining > 0:
                chunk = min(remaining, 10000)
                payload = {
                    "jsonrpc": "2.0",
                    "method": "generateIntegers",
                    "params": {
                        "apiKey": self._api_key,
                        "n": chunk,
                        "min": 0,
                        "max": 1,
                        "replacement": True,
                    },
                    "id": 1,
                }
                resp = requests.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                bits.extend(data["result"]["random"]["data"])
                remaining -= chunk
        else:
            remaining = n
            while remaining > 0:
                chunk = min(remaining, 10000)
                params: dict[str, str | int] = {
                    "num": chunk, "min": 0, "max": 1,
                    "col": 1, "base": 10, "format": "plain", "rnd": "new",
                }
                resp = requests.get(
                    "https://www.random.org/integers/",
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                vals = [int(x) for x in resp.text.strip().split()]
                bits.extend(vals)
                remaining -= chunk

        return np.array(bits[:n], dtype=np.uint8)

    def reset(self) -> None:
        """No-op -- atmospheric noise has no resettable state.

        Returns:
            None
        """
        pass


class ANUQRNGGenerator(Generator):
    """ANU Quantum Random Numbers — quantum vacuum fluctuations.

    The ANU QRNG (qrng.anu.edu.au) measures quantum vacuum
    fluctuations of the electromagnetic field in real time.
    Free API, no key required (rate-limited to 1024 numbers/request).
    Requires: network access, `requests` package.
    """

    def __init__(self, *, block_size: int = 1024) -> None:
        """Configure ANU QRNG with bytes per API request.

        Args:
            block_size: Max bytes per request (API caps at 1024).

        Raises:
            ValueError: If block_size exceeds 1024.
        """
        require_package("requests")
        _warn_intrusive(
            "ANU QRNG",
            "Will make HTTPS requests to qrng.anu.edu.au",
            "Free API, but rate-limited. Large requests may be slow.",
        )
        if block_size > 1024:
            raise ValueError(
                f"ANU API limits block_size to 1024, got {block_size}"
            )
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'ANU QRNG (Vacuum Fluctuations)'.
        """
        return "ANU QRNG (Vacuum Fluctuations)"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Fetch n random bits from the ANU QRNG API.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        import requests

        chunks: list[NDArray[np.uint8]] = []
        collected = 0
        while collected < n:
            needed_bytes = min(self._block_size, ((n - collected) + 7) // 8)
            anu_params: dict[str, str | int] = {"type": "uint8", "length": needed_bytes}
            resp = requests.get(
                "https://qrng.anu.edu.au/API/jsonI.php",
                params=anu_params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            raw = np.array(data, dtype=np.uint8)
            new_bits = np.unpackbits(raw)
            chunks.append(new_bits)
            collected += len(new_bits)

        return np.concatenate(chunks)[:n]

    def reset(self) -> None:
        """No-op -- quantum vacuum source has no resettable state.

        Returns:
            None
        """
        pass


class HotBitsGenerator(Generator):
    """HotBits — radioactive decay (Fourmilab).

    John Walker's HotBits service measures intervals between
    radioactive decays of Krypton-85 using a Geiger-Muller tube.
    Requires an API key (free tier discontinued). Max 2048 bytes per request.
    Get a key at: https://www.fourmilab.ch/hotbits/apikey_request.html
    Requires: network access, `requests` package, API key.
    """

    def __init__(self, *, api_key: str | None = None) -> None:
        """Set up HotBits client.

        Args:
            api_key: API key from fourmilab.ch (required since free tier was removed).

        Raises:
            ValueError: If no API key is provided.
        """
        if not api_key:
            raise ValueError(
                "HotBits now requires an API key. "
                "Request one at https://www.fourmilab.ch/hotbits/apikey_request.html"
            )
        require_package("requests")
        _warn_intrusive(
            "HotBits",
            "Will make HTTPS requests to fourmilab.ch",
            "Max 2048 bytes/request. Quota depends on API key tier.",
        )
        self._api_key = api_key

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'HotBits (Radioactive Decay)'.
        """
        return "HotBits (Radioactive Decay)"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Fetch n random bits from Fourmilab's radioactive decay service.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        import requests

        chunks: list[NDArray[np.uint8]] = []
        collected = 0
        while collected < n:
            nbytes = min(2048, ((n - collected) + 7) // 8)
            params: dict[str, str | int] = {"nbytes": nbytes, "fmt": "hex"}
            if self._api_key:
                params["apikey"] = self._api_key
            resp = requests.get(
                "https://www.fourmilab.ch/cgi-bin/Hotbits",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            hex_str = resp.text.strip()
            if hex_str.startswith("<!") or hex_str.startswith("<html"):
                raise RuntimeError("HotBits returned HTML — API key may be invalid or quota exhausted")
            raw = np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8)
            new_bits = np.unpackbits(raw)
            chunks.append(new_bits)
            collected += len(new_bits)

        return np.concatenate(chunks)[:n]

    def reset(self) -> None:
        """No-op -- radioactive decay has no resettable state.

        Returns:
            None
        """
        pass


# ── Hardware-based TRNGs ──────────────────────────────────────

class _RDInstructionGenerator(Generator):
    """Shared base for RDRAND and RDSEED generators.

    Handles the fallback to /dev/random and caches the file handle.
    Subclasses set _instruction and _gen_name, and override _get_hw_bits.
    """

    _instruction: str
    _gen_name: str

    def __init__(self) -> None:
        """Check CPU support and set up fallback to /dev/random if needed."""
        _check_x86_64()
        _check_cpuid_feature(self._instruction)
        self._use_fallback = False
        self._fallback_fh: IO[bytes] | None = None
        try:
            import rdrand  # noqa: F401
        except (ImportError, Exception):
            if sys.platform == "linux":
                warnings.warn(
                    f"rdrand package unavailable, falling back to /dev/random",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._use_fallback = True
                self._fallback_fh = open("/dev/random", "rb")
            else:
                raise RuntimeError(
                    f"No {self._instruction} source available on this platform"
                )

    @abstractmethod
    def _get_hw_bits(self) -> int:
        """Get 64 bits from the hardware instruction.

        Returns:
            int: 64-bit unsigned integer from hardware.
        """
        ...

    def _get_bits_64(self) -> int:
        """Get 64 bits from hardware instruction or /dev/random fallback.

        Returns:
            int: 64-bit unsigned integer.
        """
        if self._use_fallback:
            assert self._fallback_fh is not None
            return int(struct.unpack("<Q", self._fallback_fh.read(8))[0])
        return self._get_hw_bits()

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: The generator name set by the subclass.
        """
        return self._gen_name

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits by pulling 64-bit words from hardware and unpacking.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        chunks: list[NDArray[np.uint8]] = []
        collected = 0
        while collected < n:
            val = self._get_bits_64()
            word_bits = unpack_uint64(val, min(64, n - collected))
            chunks.append(word_bits)
            collected += len(word_bits)
        return np.concatenate(chunks)[:n]

    def reset(self) -> None:
        """No-op -- hardware entropy has no resettable state.

        Returns:
            None
        """
        pass

    def __del__(self) -> None:
        """Close the /dev/random fallback file handle if open.

        Returns:
            None
        """
        fh = getattr(self, "_fallback_fh", None)
        if fh is not None:
            fh.close()


class RDRANDGenerator(_RDInstructionGenerator):
    """Intel/AMD RDRAND CPU instruction.

    Uses the on-chip digital random number generator present in
    Intel Ivy Bridge+ and AMD Zen+ CPUs. Reads from a conditioned
    entropy source (thermal noise on-die).
    Requires: x86_64 CPU with RDRAND support.
    """
    _instruction = "RDRAND"
    _gen_name = "RDRAND (CPU Hardware)"

    def _get_hw_bits(self) -> int:
        """Read 64 bits from the RDRAND instruction.

        Returns:
            int: 64-bit unsigned integer from RDRAND.
        """
        import rdrand
        return int(rdrand.rdrand_get_bits(64))


class RDSEEDGenerator(_RDInstructionGenerator):
    """Intel/AMD RDSEED CPU instruction.

    Unlike RDRAND, RDSEED provides access to the raw entropy source
    without deterministic conditioning. Slower but provides true
    entropy suitable for seeding other generators.
    Requires: x86_64 CPU with RDSEED support (Broadwell+/Zen+).
    """
    _instruction = "RDSEED"
    _gen_name = "RDSEED (CPU Entropy)"

    def _get_hw_bits(self) -> int:
        """Read 64 bits from the RDSEED instruction.

        Returns:
            int: 64-bit unsigned integer from RDSEED.
        """
        import rdrand
        return int(rdrand.rdseed_get_bits(64))


class HWRNGDevice(Generator):
    """Linux /dev/hwrng — kernel hardware RNG.

    Reads from the kernel's hardware RNG interface, which aggregates
    entropy from available hardware sources (TPM, RDRAND, VirtIO, etc.).
    Requires: Linux, read permission on the device.
    """

    def __init__(self, *, device_path: Path = Path("/dev/hwrng")) -> None:
        """Validate the hwrng device exists and is readable.

        Args:
            device_path: Path to the hardware RNG device file.

        Raises:
            RuntimeError: If not running on Linux.
            FileNotFoundError: If the device doesn't exist.
            PermissionError: If the device isn't readable.
        """
        _check_linux()
        _check_device_readable(device_path)
        self._device_path = device_path

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Name like 'HW RNG (/dev/hwrng)'.
        """
        return f"HW RNG ({self._device_path})"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Read n bits from the kernel hardware RNG device.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        nbytes = (n + 7) // 8
        with open(self._device_path, "rb") as f:
            raw = f.read(nbytes)
        return bytes_to_bits(raw, n)

    def reset(self) -> None:
        """No-op -- hardware device has no resettable state.

        Returns:
            None
        """
        pass


# ── Environmental noise TRNGs ─────────────────────────────────

class TimingJitterGenerator(Generator):
    """CPU timing jitter — a la HAVEGE / jitterentropy.

    Measures variations in CPU instruction timing caused by caches,
    branch prediction, TLB, and other microarchitectural effects.
    No special hardware needed, but quality is platform-dependent.

    Warning: VMs with coarse timer resolution may produce low-quality
    output. Not suitable as a sole entropy source without validation.
    """

    def __init__(self, *, samples_per_bit: int = 64) -> None:
        """Configure jitter sampling rate.

        Args:
            samples_per_bit: Number of timing measurements per output bit.

        Raises:
            ValueError: If samples_per_bit < 1.
        """
        if samples_per_bit < 1:
            raise ValueError(
                f"samples_per_bit must be >= 1, got {samples_per_bit}"
            )
        _warn_intrusive(
            "TimingJitter",
            "Will busy-loop the CPU to measure timing variations",
            "May cause high CPU usage and degrade performance of other processes.",
        )
        self._samples_per_bit = samples_per_bit

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'Timing Jitter (CPU)'.
        """
        return "Timing Jitter (CPU)"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits by measuring CPU timing jitter via SHA-256 hashing.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        bits = np.empty(n, dtype=np.uint8)
        counter = 0
        for i in range(n):
            deltas = 0
            for _ in range(self._samples_per_bit):
                t0 = time.perf_counter_ns()
                hashlib.sha256(counter.to_bytes(8, "little")).digest()
                counter += 1
                t1 = time.perf_counter_ns()
                deltas += (t1 - t0)
            bits[i] = deltas & 1
        return bits

    def reset(self) -> None:
        """No-op -- timing jitter is non-deterministic.

        Returns:
            None
        """
        pass


class AudioNoiseGenerator(Generator):
    """Microphone thermal/environmental noise.

    Samples the least significant bits of audio input, which are
    dominated by thermal noise in the ADC. Requires a connected
    microphone.
    Requires: `sounddevice` package, functioning audio input device.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 44100,
        bits_per_sample: int = 1,
    ) -> None:
        """Set up audio noise source.

        Args:
            sample_rate: Audio sample rate in Hz.
            bits_per_sample: Number of LSBs to extract per audio sample (1-16).

        Raises:
            RuntimeError: If no audio input devices are found.
            ValueError: If bits_per_sample is outside [1, 16].
        """
        require_package("sounddevice")
        import sounddevice as sd

        devices = sd.query_devices()
        input_devices = [d for d in devices if d["max_input_channels"] > 0]
        if not input_devices:
            raise RuntimeError(
                "No audio input devices found. Connect a microphone "
                "and ensure it is recognised by the OS."
            )

        _warn_intrusive(
            "AudioNoise",
            "Will access the microphone to record audio samples",
            "The OS may prompt for microphone permission. "
            "Ensure no sensitive audio is being captured.",
        )

        if bits_per_sample < 1 or bits_per_sample > 16:
            raise ValueError(
                f"bits_per_sample must be in [1, 16], got {bits_per_sample}"
            )

        self._sample_rate = sample_rate
        self._bits_per_sample = bits_per_sample

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'Audio Noise (Microphone)'.
        """
        return "Audio Noise (Microphone)"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Record audio and extract n bits from the LSBs of each sample.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        import sounddevice as sd

        samples_needed = (n // self._bits_per_sample) + 1
        recording = sd.rec(
            samples_needed,
            samplerate=self._sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()

        raw = recording.flatten().astype(np.int16)
        mask = (1 << self._bits_per_sample) - 1
        lsbs = raw & mask

        bits: list[int] = []
        for val in lsbs:
            for b in range(self._bits_per_sample):
                bits.append((int(val) >> b) & 1)
                if len(bits) >= n:
                    break
            if len(bits) >= n:
                break

        return np.array(bits[:n], dtype=np.uint8)

    def reset(self) -> None:
        """No-op -- microphone noise is non-deterministic.

        Returns:
            None
        """
        pass


class CameraNoiseLavaRand(Generator):
    """Camera sensor noise — LavaRand-style.

    Inspired by Cloudflare's LavaRand and Silicon Graphics' original
    lava lamp wall. Captures images from a camera/webcam and hashes
    the raw sensor noise.
    Requires: `opencv-python` package, functioning camera.
    """

    def __init__(self, *, camera_index: int = 0) -> None:
        """Validate camera access at the given index.

        Args:
            camera_index: OpenCV camera device index.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        require_package("cv2", "opencv-python")
        import cv2

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Cannot open camera at index {camera_index}. Ensure a "
                f"camera is connected and not in use by another application."
            )
        cap.release()

        _warn_intrusive(
            "CameraNoiseLavaRand",
            "Will access the camera to capture image frames",
            "The OS may prompt for camera permission. "
            "The camera LED will activate during generation.",
        )

        self._camera_index = camera_index

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'Camera Noise (LavaRand)'.
        """
        return "Camera Noise (LavaRand)"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Capture frames, SHA-256 hash each, and extract n bits total.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        import cv2

        chunks: list[NDArray[np.uint8]] = []
        collected = 0
        cap = cv2.VideoCapture(self._camera_index)
        try:
            while collected < n:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to capture frame from camera")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h = hashlib.sha256(gray.tobytes()).digest()
                hash_bits = bytes_to_bits(h, min(256, n - collected))
                chunks.append(hash_bits)
                collected += len(hash_bits)
        finally:
            cap.release()

        return np.concatenate(chunks)[:n]

    def reset(self) -> None:
        """No-op -- camera noise is non-deterministic.

        Returns:
            None
        """
        pass
