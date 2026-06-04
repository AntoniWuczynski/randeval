# randeval

A Python package for generating and evaluating random number sequences. Built for a masters thesis comparing classical PRNGs, CSPRNGs, TRNGs, and quantum random number generators.

- **32 generators** across 4 categories (classical, CSPRNG, TRNG, quantum)
- **68 statistical tests** (NIST SP 800-22, Dieharder, NIST SP 800-90B, information-theoretic, distribution, autocorrelation, novel)
- **27 randomness extractors** across 6 families (debiasing, universal hashing, computational, practical, adaptive, CryptoMite)
- Strongly typed, PEP 561 compliant (`py.typed`)
- External sequences welcome — wrap any `list[int]`, `NDArray`, or `bytes`

## Install

```bash
cd randeval

# Core: classical + CSPRNG generators and the full statistical-test battery
uv sync

# Optional extras
uv sync --extra quantum      # Qiskit + IBM Quantum generators
uv sync --extra extractors   # CryptoMite extractor constructions
uv sync --extra rendering    # Mitsuba 3 + FLIP (rendering case study)
uv sync --extra dev          # pytest, mypy, nistrng (tests + validation)

# Everything at once
uv sync --all-extras

# Install from PyPI
pip install randeval                                      # core only
pip install "randeval[quantum,extractors,rendering,dev]"  # everything

# Editable install into another project
uv pip install -e .
uv pip install -e ".[quantum,extractors,rendering]"
```

## Package Structure

```
src/randeval/
├── sequence.py              # RandomSequence — core data object
├── generators/
│   ├── base.py              # Generator ABC
│   ├── classical.py         # 11 classical PRNGs
│   ├── csprng.py            # 6 CSPRNGs
│   ├── trng.py              # 9 TRNGs (API, hardware, environmental)
│   └── quantum.py           # 6 quantum generators (optional)
├── extractors/
│   ├── base.py              # Extractor ABC
│   ├── debiasing.py         # Von Neumann, Peres, Elias, AMLS
│   ├── hashing.py           # 5 universal-hash extractors
│   ├── computational.py     # HMAC, SHA-256 conditioner
│   ├── practical.py         # 9 XOR / decimation-style extractors
│   ├── adaptive.py          # arithmetic coding, min-entropy, fuzzy
│   └── cryptomite_ext.py    # Toeplitz, Circulant, Dodis, Trevisan
├── tests_statistical/
│   ├── base.py              # StatisticalTest ABC, TestResult, Verdict
│   ├── nist.py              # 16 NIST SP 800-22 tests
│   ├── dieharder.py         # 18 Dieharder / Marsaglia tests
│   ├── sp800_90b.py         # 12 NIST SP 800-90B entropy estimators
│   ├── entropy.py           # 8 information-theoretic measures
│   ├── distribution.py      # 6 general statistical tests
│   ├── autocorrelation.py   # Lag-based autocorrelation
│   └── novel.py             # 7 novel tests
└── py.typed
```

## Generators (32)

### Classical PRNGs — `generators/classical.py` (11)

| Class                | Algorithm                      | Notes                                       |
| -------------------- | ------------------------------ | ------------------------------------------- |
| `LCG`                | Linear Congruential            | glibc defaults, configurable bit extraction |
| `MersenneTwister`    | MT19937                        | Python's `random` default                   |
| `PCG64`              | Permuted Congruential          | NumPy's default                             |
| `Xorshift128Plus`    | Xorshift128+                   | Used in V8/WebKit JS engines                |
| `Xoshiro256StarStar` | xoshiro256\*\*                 | Blackman & Vigna, passes BigCrush           |
| `LFSR`               | Linear Feedback Shift Register | Configurable taps/width                     |
| `MiddleSquare`       | Von Neumann (1946)             | Historical, degenerates quickly             |
| `MiddleSquareWeyl`   | Middle Square Weyl Sequence    | Widynski (2020), fixes original             |
| `Philox`             | Philox4x64 counter-based       | Used in NumPy/TF/JAX, parallelisable        |
| `SFC64`              | Small Fast Chaotic 64          | NumPy BitGenerator option                   |
| `WichmannHill`       | Three combined LCGs            | Python's pre-MT default                     |

### CSPRNGs — `generators/csprng.py` (6)

| Class          | Algorithm              | Notes                                   |
| -------------- | ---------------------- | --------------------------------------- |
| `SystemRandom` | OS entropy pool        | `os.urandom` / `secrets`                |
| `ChaCha20`     | ChaCha20 stream cipher | Requires `cryptography`                 |
| `AESCTR_DRBG`  | AES-256-CTR DRBG       | Requires `cryptography`                 |
| `HMAC_DRBG`    | HMAC-SHA256 DRBG       | stdlib only                             |
| `Hash_DRBG`    | SHA-256 DRBG           | stdlib only                             |
| `BlumBlumShub` | x² mod pq              | Validates p ≡ q ≡ 3 mod 4, coprime seed |

### TRNGs — `generators/trng.py` (9)

| Class                   | Entropy Source                  | Requirements              |
| ----------------------- | ------------------------------- | ------------------------- |
| `RandomOrgGenerator`    | Atmospheric noise               | `requests`, network       |
| `ANUQRNGGenerator`      | Quantum vacuum fluctuations     | `requests`, network       |
| `HotBitsGenerator`      | Radioactive decay (Kr-85)       | `requests`, network       |
| `RDRANDGenerator`       | CPU thermal noise (conditioned) | x86_64 only               |
| `RDSEEDGenerator`       | CPU raw entropy                 | x86_64 (Broadwell+/Zen+)  |
| `HWRNGDevice`           | Linux `/dev/hwrng`              | Linux, read permission    |
| `TimingJitterGenerator` | CPU timing jitter               | None (quality varies)     |
| `AudioNoiseGenerator`   | Microphone ADC noise            | `sounddevice`, microphone |
| `CameraNoiseLavaRand`   | Camera sensor noise             | `opencv-python`, camera   |

### Quantum — `generators/quantum.py` (6, optional)

| Class                        | Circuit            | Notes                                 |
| ---------------------------- | ------------------ | ------------------------------------- |
| `QiskitSimulator`            | 1-qubit Hadamard   | StatevectorSampler, 1 bit/shot        |
| `IBMQuantumBackend`          | 1-qubit Hadamard   | Real hardware, cloud queue            |
| `MultiQubitHadamard`         | N-qubit parallel H | N bits/shot, sim or real              |
| `EntanglementBasedQRNG`      | Bell pairs (EPR)   | CHSH-verifiable quantumness           |
| `QuantumPhaseEstimationQRNG` | QPE circuit        | Multi-bit output per shot             |
| `RandomRotationQRNG`         | Ry(θ) + measure    | Parametric bias for extractor testing |

> Quantum generators require `pip install randeval[quantum]` (Qiskit + IBM Runtime).

## Statistical Tests (68)

### NIST SP 800-22 — `tests_statistical/nist.py` (16)

Frequency, Block Frequency, Runs, Longest Run of Ones, Binary Matrix Rank,
DFT Spectral, Non-overlapping Template, Overlapping Template, Maurer's Universal,
Linear Complexity, Serial, Approximate Entropy, Cumulative Sums (fwd + bwd),
Random Excursions, Random Excursions Variant.

### Dieharder / Marsaglia — `tests_statistical/dieharder.py` (18)

Birthday Spacings, Overlapping Permutations, Parking Lot, Minimum Distance,
3D Spheres, Squeeze, Overlapping Sums, Craps, GCD, Gorilla, Coupon Collector,
Gap, Poker, Collision, Bitstream, DNA, Count-the-1s (stream + byte).

### NIST SP 800-90B — `tests_statistical/sp800_90b.py` (12)

Most Common Value, Collision, Markov, Compression, t-Tuple, LRS,
Repetition Count, Adaptive Proportion, and the MultiMCW, Lag, MultiMMC
and LZ78Y predictors.

### Information-Theoretic — `tests_statistical/entropy.py` (8)

Shannon Entropy, Min-Entropy, Rényi Entropy, Compression Ratio,
Lempel-Ziv Complexity, Conditional Entropy, Mutual Information,
Permutation Entropy.

### Distribution — `tests_statistical/distribution.py` (6)

Chi-Squared Uniformity, Kolmogorov-Smirnov, Anderson-Darling,
Wald-Wolfowitz Runs, Mann-Kendall Trend, Turning Point.

### Autocorrelation — `tests_statistical/autocorrelation.py` (1)

Lag-based autocorrelation test with configurable lag range.

### Novel — `tests_statistical/novel.py` (7)

Running bias, bit-pattern spatial, weight distribution, close pairs,
max-of-t, successive differences, byte runs.

### Battery helpers

```python
from randeval.tests_statistical import (
    nist_battery,        # 16 NIST SP 800-22 tests
    dieharder_battery,   # 18 Dieharder tests
    sp800_90b_battery,   # 12 NIST SP 800-90B estimators
    entropy_battery,     # 8 entropy tests
    distribution_battery,# 6 distribution tests
    novel_battery,       # 7 novel tests
    full_battery,        # all 68 above combined (+ autocorrelation)
)
```

## Extractors

27 extractors across 6 families:

| Family            | Count | Examples                                                 |
| ----------------- | ----- | -------------------------------------------------------- |
| Debiasing         | 4     | Von Neumann, Peres, Elias, AMLS                          |
| Universal hashing | 5     | Toeplitz, linear, inner-product, LHL, polynomial         |
| Computational     | 2     | HMAC, SHA-256 conditioner                                |
| Practical         | 9     | XOR, block parity, bit decimation, CRC32, subsampling, … |
| Adaptive          | 3     | arithmetic coding, min-entropy, fuzzy extractor          |
| CryptoMite        | 4     | Toeplitz, Circulant, Dodis, Trevisan                     |

Implement the `Extractor` ABC to add more.

> The CryptoMite constructions (Toeplitz, Circulant, Dodis, Trevisan) require `uv sync --extra extractors`.

## Intrusive Generator Warnings

Generators that access peripherals, network, or cloud services emit a
`UserWarning` on instantiation:

| Generator               | Warning                                        |
| ----------------------- | ---------------------------------------------- |
| `RandomOrgGenerator`    | HTTPS to api.random.org                        |
| `ANUQRNGGenerator`      | HTTPS to qrng.anu.edu.au                       |
| `HotBitsGenerator`      | HTTPS to fourmilab.ch                          |
| `TimingJitterGenerator` | CPU busy-loop                                  |
| `AudioNoiseGenerator`   | Microphone access                              |
| `CameraNoiseLavaRand`   | Camera access, LED activates                   |
| `IBMQuantumBackend`     | IBM cloud queue, token transmitted             |
| `MultiQubitHadamard`    | IBM cloud queue (when `use_real_backend=True`) |

Suppress with `warnings.filterwarnings("ignore", category=UserWarning)`.

## Quick Usage

```python
from randeval import RandomSequence
from randeval.generators.classical import LCG
from randeval.tests_statistical import full_battery, nist_battery
from randeval.tests_statistical.nist import FrequencyTest

# Generate from a generator
seq = RandomSequence.from_generator(LCG(seed=42), n=10_000)

# Or wrap externally-produced bits
import numpy as np
seq = RandomSequence(np.array([0, 1, 1, 0, 1], dtype=np.uint8))

# Or from raw bytes
seq = RandomSequence(b"\xaf\x3c\x01")

# Run a single test
result = seq.test(FrequencyTest())
print(result)  # TestResult('NIST 1: Frequency (Monobit)', p=0.7342, pass)

# Run the full NIST battery
results = seq.test_suite(nist_battery())

# Run all 68 battery tests
results = seq.test_suite(full_battery())

# Apply an extractor then re-test
from randeval.extractors.von_neumann import VonNeumannExtractor
clean = seq.extract(VonNeumannExtractor())
clean_results = clean.test_suite(nist_battery())
```

## Extending

Implement `Generator`, `Extractor`, or `StatisticalTest` to add new components:

```python
from randeval.generators.base import Generator
import numpy as np
from numpy.typing import NDArray

class MyGenerator(Generator):
    @property
    def name(self) -> str:
        return "MyGenerator"

    def generate(self, n: int) -> NDArray[np.uint8]:
        # Return array of n bits (0s and 1s)
        ...
```

## Testing & Validation

```bash
uv run --extra dev pytest tests -q          # full suite (extractors + statistical-test validation)
uv run --extra dev pytest tests/statistical -q   # just the statistical-test validation
uv run --extra dev mypy --strict src/       # type-check (clean under strict mode)
```

`tests/statistical/` validates the statistical tests against ground truth:

- **`test_nist_kat.py`** — known-answer tests against NIST SP 800-22 Rev 1a
  worked examples (the 100-bit pi sequence and the vendored `data.e`/`data.pi`
  reference files in `fixtures/`).
- **`test_null_calibration.py`** — confirms tests reject good random data at
  ~alpha and produce ~Uniform(0,1) p-values.
- **`test_cross_impl.py`** — cross-checks against `nistrng` (optional dev dep).
- **`test_sanity.py`** — good data passes the battery, broken data is flagged.
- **`run_calibration.py`** — heavier 200-stream tiered calibration sweep
  (`uv run --extra dev python -m tests.statistical.run_calibration`).
