"""Quantum random number generators.

Requires the `quantum` extra: pip install randeval[quantum]
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from ._utils import require_package
from .base import Generator


def _resolve_ibm_token(token: str | None) -> str:
    """Resolve an IBM Quantum API token from argument, env, or .env file.

    Args:
        token: Explicit token string, or None to search env/dotenv.

    Returns:
        str: The resolved API token.

    Raises:
        ValueError: If no token is found anywhere.
    """
    if token:
        return token
    import os
    t = os.environ.get("IBM_TOKEN")
    if t:
        return t
    try:
        from dotenv import load_dotenv
        load_dotenv()
        t = os.environ.get("IBM_TOKEN")
        if t:
            return t
    except ImportError:
        pass
    raise ValueError("No IBM Quantum token found")


def _run_sampler_loop(
    buffer: list[int],
    sampler: Any,
    qc: Any,
    n: int,
    shots_per_circuit: int,
    extract_fn: Callable[[list[str]], list[int]],
) -> tuple[NDArray[np.uint8], list[int]]:
    """Drain buffer, run sampler in batches until n bits collected.

    Args:
        buffer: Leftover bits from a previous call (will be cleared).
        sampler: Qiskit sampler primitive instance.
        qc: Compiled quantum circuit to sample from.
        n: Total number of bits needed.
        shots_per_circuit: Max shots per sampler invocation.
        extract_fn: Converts list of bitstrings to list of ints.

    Returns:
        tuple[NDArray[np.uint8], list[int]]: Array of n bits and leftover buffer.
    """
    bits = list(buffer)
    buffer.clear()
    while len(bits) < n:
        shots = min(shots_per_circuit, n - len(bits))
        job = sampler.run([qc], shots=shots)
        result = job.result()
        bitstrings = result[0].data.c.get_bitstrings()
        bits.extend(extract_fn(bitstrings))
    leftover = bits[n:]
    return np.array(bits[:n], dtype=np.uint8), leftover


class QiskitSimulator(Generator):
    """Single-qubit Hadamard circuit on the Qiskit StatevectorSampler.

    One bit per shot. Simple but slow for large n.
    """

    is_simulated = True

    def __init__(self, *, shots_per_circuit: int = 8192) -> None:
        """Configure the simulator sampler.

        Args:
            shots_per_circuit: Max shots per sampler invocation.
        """
        require_package("qiskit", "randeval[quantum]")
        self._shots_per_circuit = shots_per_circuit
        self._buffer: list[int] = []
        self._qc: Any | None = None
        self._sampler: Any | None = None

    def _ensure_circuit(self) -> None:
        """Lazily build the single-qubit H+measure circuit and sampler.

        Returns:
            None
        """
        if self._qc is not None:
            return
        from qiskit import QuantumCircuit
        from qiskit.primitives import StatevectorSampler

        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        self._qc = qc
        self._sampler = StatevectorSampler()

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'QiskitSimulator'.
        """
        return "QiskitSimulator"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits via simulated Hadamard measurement, one bit per shot.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        self._ensure_circuit()
        arr, self._buffer = _run_sampler_loop(
            self._buffer, self._sampler, self._qc, n,
            self._shots_per_circuit,
            lambda bss: [int(b) for b in bss],
        )
        return arr

    def reset(self) -> None:
        """Clear the leftover bit buffer.

        Returns:
            None
        """
        self._buffer.clear()


class IBMQuantumBackend(Generator):
    """Real IBM Quantum hardware via qiskit-ibm-runtime.

    Requires an IBM Quantum API token (loaded from env or passed directly).
    Subject to queue times, gate errors, and readout noise.
    """

    def __init__(
        self,
        *,
        backend_name: str = "ibm_brisbane",
        token: str | None = None,
        shots_per_circuit: int = 8192,
    ) -> None:
        """Configure for a real IBM backend.

        Args:
            backend_name: IBM Quantum backend to target.
            token: API token. Resolved from env/dotenv if not given.
            shots_per_circuit: Max shots per job submission.
        """
        require_package("qiskit", "randeval[quantum]")
        import warnings
        warnings.warn(
            "[IBMQuantumBackend] Will submit jobs to IBM Quantum cloud. "
            "Jobs enter a queue and may take minutes to hours depending on "
            "backend load. API token will be transmitted to IBM servers.",
            UserWarning,
            stacklevel=2,
        )
        self._backend_name = backend_name
        self._token = _resolve_ibm_token(token)
        self._shots_per_circuit = shots_per_circuit
        self._buffer: list[int] = []

        self._service: object | None = None
        self._backend: object | None = None
        self._sampler: object | None = None
        self._transpiled_qc: Any | None = None

    def _ensure_backend(self) -> None:
        """Lazily initialise the IBM runtime service, backend, and sampler.

        Returns:
            None
        """
        if self._sampler is not None:
            return
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

        self._service = QiskitRuntimeService(
            channel="ibm_quantum_platform", token=self._token,
        )
        if self._backend_name:
            self._backend = self._service.backend(self._backend_name)  # type: ignore[union-attr]
        else:
            self._backend = self._service.least_busy(operational=True)  # type: ignore[union-attr]
        self._sampler = SamplerV2(self._backend)

    def _ensure_circuit(self) -> None:
        """Lazily build and transpile the H+measure circuit for the real backend.

        Returns:
            None
        """
        if self._transpiled_qc is not None:
            return
        from qiskit import QuantumCircuit, transpile

        self._ensure_backend()
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.measure(0, 0)
        self._transpiled_qc = transpile(qc, self._backend)

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Name like 'IBMQuantum(ibm_brisbane)'.
        """
        return f"IBMQuantum({self._backend_name})"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Submit jobs to IBM hardware and collect n bits (may queue).

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        self._ensure_circuit()

        bits = list(self._buffer)
        self._buffer.clear()

        while len(bits) < n:
            shots = min(self._shots_per_circuit, n - len(bits))
            job = self._sampler.run([self._transpiled_qc], shots=shots)  # type: ignore[union-attr]
            result = job.result()
            pub_result = result[0]
            try:
                bitstrings = pub_result.data.c.get_bitstrings()
                bits.extend(int(b) for b in bitstrings)
            except AttributeError:
                counts = pub_result.data.c.get_counts()
                for bitval, count in counts.items():
                    bits.extend([int(bitval)] * count)

        self._buffer = bits[n:]
        return np.array(bits[:n], dtype=np.uint8)

    def reset(self) -> None:
        """Clear the leftover bit buffer.

        Returns:
            None
        """
        self._buffer.clear()


class MultiQubitHadamard(Generator):
    """N-qubit parallel Hadamard circuit.

    Generates `num_qubits` bits per shot, much faster than single-qubit
    for large sequences. Default 8 qubits = 1 byte per shot.
    """

    is_simulated = True

    def __init__(
        self,
        *,
        num_qubits: int = 8,
        shots_per_circuit: int = 8192,
        use_real_backend: bool = False,
        backend_name: str = "ibm_brisbane",
        token: str | None = None,
    ) -> None:
        """Configure multi-qubit Hadamard generator.

        Args:
            num_qubits: Number of qubits (bits per shot).
            shots_per_circuit: Max shots per sampler invocation.
            use_real_backend: If True, submit to IBM hardware instead of simulator.
            backend_name: IBM Quantum backend name (only if use_real_backend).
            token: IBM Quantum API token (only if use_real_backend).
        """
        require_package("qiskit", "randeval[quantum]")
        if use_real_backend:
            import warnings
            warnings.warn(
                f"[MultiQubitHadamard] use_real_backend=True — will submit "
                f"jobs to IBM Quantum cloud ({backend_name}). "
                f"Jobs enter a queue and may take minutes to hours.",
                UserWarning,
                stacklevel=2,
            )
        self._num_qubits = num_qubits
        self._shots_per_circuit = shots_per_circuit
        self._use_real_backend = use_real_backend
        self._backend_name = backend_name
        self._token = token
        self._buffer: list[int] = []

        self._service: object | None = None
        self._backend: object | None = None
        self._sampler: Any | None = None
        self._qc: Any | None = None

    def _ensure_backend(self) -> None:
        """Lazily create the sampler -- real IBM backend or local simulator.

        Returns:
            None
        """
        if self._sampler is not None:
            return
        if self._use_real_backend:
            from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

            token = _resolve_ibm_token(self._token)
            self._service = QiskitRuntimeService(
                channel="ibm_quantum_platform", token=token,
            )
            self._backend = self._service.backend(self._backend_name)  # type: ignore[union-attr]
            self._sampler = SamplerV2(self._backend)
        else:
            from qiskit.primitives import StatevectorSampler
            self._sampler = StatevectorSampler()

    def _ensure_circuit(self) -> None:
        """Lazily build the N-qubit all-Hadamard circuit, transpile if real backend.

        Returns:
            None
        """
        if self._qc is not None:
            return
        from qiskit import QuantumCircuit

        self._ensure_backend()
        qc = QuantumCircuit(self._num_qubits, self._num_qubits)
        qc.h(range(self._num_qubits))
        qc.measure(range(self._num_qubits), range(self._num_qubits))
        if self._use_real_backend:
            from qiskit import transpile
            qc = transpile(qc, self._backend)
        self._qc = qc

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Name like 'MultiQubitHadamard(8q, simulator)'.
        """
        backend = self._backend_name if self._use_real_backend else "simulator"
        return f"MultiQubitHadamard({self._num_qubits}q, {backend})"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits, getting num_qubits bits per shot.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        self._ensure_circuit()

        bits = list(self._buffer)
        self._buffer.clear()

        shots_needed = math.ceil(n / self._num_qubits)
        collected = 0
        while collected < shots_needed:
            batch = min(self._shots_per_circuit, shots_needed - collected)
            job = self._sampler.run([self._qc], shots=batch)  # type: ignore[union-attr]
            result = job.result()
            bitstrings = result[0].data.c.get_bitstrings()
            for bs in bitstrings:
                bits.extend(int(ch) for ch in bs)
            collected += batch

        self._buffer = bits[n:]
        return np.array(bits[:n], dtype=np.uint8)

    def reset(self) -> None:
        """Clear the leftover bit buffer.

        Returns:
            None
        """
        self._buffer.clear()


class EntanglementBasedQRNG(Generator):
    """Bell-pair (EPR) based quantum RNG.

    Creates entangled pairs and measures both qubits. Can be used to
    verify quantumness via CHSH inequality violation.
    Simulated only — real entanglement-based QRNG needs specialised hardware.
    """

    is_simulated = True

    def __init__(self, *, shots_per_circuit: int = 8192) -> None:
        """Configure the Bell-pair sampler.

        Args:
            shots_per_circuit: Max shots per sampler invocation.
        """
        require_package("qiskit", "randeval[quantum]")
        self._shots_per_circuit = shots_per_circuit
        self._buffer: list[int] = []
        self._qc: Any | None = None
        self._sampler: Any | None = None

    def _ensure_circuit(self) -> None:
        """Lazily build the Bell-pair (H + CNOT) circuit and sampler.

        Returns:
            None
        """
        if self._qc is not None:
            return
        from qiskit import QuantumCircuit
        from qiskit.primitives import StatevectorSampler

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        self._qc = qc
        self._sampler = StatevectorSampler()

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Always 'EntanglementBasedQRNG'.
        """
        return "EntanglementBasedQRNG"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits from Bell-pair measurements (one bit per shot).

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        self._ensure_circuit()
        arr, self._buffer = _run_sampler_loop(
            self._buffer, self._sampler, self._qc, n,
            self._shots_per_circuit,
            lambda bss: [int(bs[-1]) for bs in bss],
        )
        return arr

    def reset(self) -> None:
        """Clear the leftover bit buffer.

        Returns:
            None
        """
        self._buffer.clear()


class QuantumPhaseEstimationQRNG(Generator):
    """Randomness from quantum phase estimation.

    Applies QPE with |+> on the target qubit, producing structured
    (non-uniform) output concentrated on eigenvalue-encoding bitstrings.
    Useful for studying randomness extraction from biased quantum sources.
    Higher qubit count -> more bits per shot.
    """

    is_simulated = True

    def __init__(
        self,
        *,
        num_counting_qubits: int = 4,
        shots_per_circuit: int = 8192,
    ) -> None:
        """Configure QPE-based generator.

        Args:
            num_counting_qubits: Number of counting qubits (bits per shot).
            shots_per_circuit: Max shots per sampler invocation.
        """
        require_package("qiskit", "randeval[quantum]")
        self._num_counting_qubits = num_counting_qubits
        self._shots_per_circuit = shots_per_circuit
        self._buffer: list[int] = []
        self._qc: Any | None = None
        self._sampler: Any | None = None

    def _ensure_circuit(self) -> None:
        """Lazily build the QPE circuit with inverse QFT on counting register.

        Returns:
            None
        """
        if self._qc is not None:
            return
        from qiskit import QuantumCircuit
        from qiskit.primitives import StatevectorSampler

        nq = self._num_counting_qubits
        total_qubits = nq + 1
        qc = QuantumCircuit(total_qubits, nq)

        qc.h(nq)
        for k in range(nq):
            theta = math.pi / (2 ** k)
            qc.cp(theta, k, nq)

        # inverse QFT on counting register
        for i in range(nq // 2):
            qc.swap(i, nq - 1 - i)
        for i in range(nq - 1, -1, -1):
            qc.h(i)
            for j in range(i - 1, -1, -1):
                qc.cp(-math.pi / (2 ** (i - j)), j, i)

        qc.measure(range(nq), range(nq))
        self._qc = qc
        self._sampler = StatevectorSampler()

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Name like 'QPE-QRNG(4q)'.
        """
        return f"QPE-QRNG({self._num_counting_qubits}q)"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n bits from QPE measurements, num_counting_qubits bits per shot.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        self._ensure_circuit()
        nq = self._num_counting_qubits

        bits = list(self._buffer)
        self._buffer.clear()

        shots_needed = math.ceil(n / nq)
        collected = 0
        while collected < shots_needed:
            batch = min(self._shots_per_circuit, shots_needed - collected)
            job = self._sampler.run([self._qc], shots=batch)  # type: ignore[union-attr]
            result = job.result()
            bitstrings = result[0].data.c.get_bitstrings()
            for bs in bitstrings:
                bits.extend(int(ch) for ch in bs)
            collected += batch

        self._buffer = bits[n:]
        return np.array(bits[:n], dtype=np.uint8)

    def reset(self) -> None:
        """Clear the leftover bit buffer.

        Returns:
            None
        """
        self._buffer.clear()


class RandomRotationQRNG(Generator):
    """Applies a random rotation angle before measurement.

    Uses parametric Ry(theta) gates with varying angles to explore
    different bias points. Useful for studying extractor effectiveness.
    """

    is_simulated = True

    def __init__(
        self,
        *,
        theta: float = 0.7854,
        shots_per_circuit: int = 8192,
    ) -> None:
        """Configure with Ry rotation angle and shots per run.

        Args:
            theta: Ry rotation angle in radians.
            shots_per_circuit: Max shots per sampler invocation.
        """
        require_package("qiskit", "randeval[quantum]")
        self._theta = theta
        self._shots_per_circuit = shots_per_circuit
        self._buffer: list[int] = []
        self._qc: Any | None = None
        self._sampler: Any | None = None

    def _ensure_circuit(self) -> None:
        """Lazily build the Ry(theta) + measure circuit and sampler.

        Returns:
            None
        """
        if self._qc is not None:
            return
        from qiskit import QuantumCircuit
        from qiskit.primitives import StatevectorSampler

        qc = QuantumCircuit(1, 1)
        qc.ry(self._theta, 0)
        qc.measure(0, 0)
        self._qc = qc
        self._sampler = StatevectorSampler()

    @property
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Name like 'RandomRotationQRNG(theta=0.7854)'.
        """
        return f"RandomRotationQRNG(\u03b8={self._theta:.4f})"

    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n biased bits via Ry rotation measurement.

        Args:
            n: Number of random bits to generate.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        self._ensure_circuit()
        arr, self._buffer = _run_sampler_loop(
            self._buffer, self._sampler, self._qc, n,
            self._shots_per_circuit,
            lambda bss: [int(b) for b in bss],
        )
        return arr

    def reset(self) -> None:
        """Clear the leftover bit buffer.

        Returns:
            None
        """
        self._buffer.clear()
