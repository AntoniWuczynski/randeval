"""Random number generators — classical, CSPRNG, TRNG, and quantum."""

from .base import Generator

# classical
from .classical import (
    LCG,
    MersenneTwister,
    PCG64,
    Philox,
    SFC64,
    Xorshift128Plus,
    Xoshiro256StarStar,
    LFSR,
    MiddleSquare,
    MiddleSquareWeyl,
    WichmannHill,
)

# cryptographically secure
from .csprng import (
    SystemRandom,
    ChaCha20,
    AESCTR_DRBG,
    HMAC_DRBG,
    Hash_DRBG,
    BlumBlumShub,
)

# true random (hardware / environmental / API)
from .trng import (
    RandomOrgGenerator,
    ANUQRNGGenerator,
    HotBitsGenerator,
    RDRANDGenerator,
    RDSEEDGenerator,
    HWRNGDevice,
    TimingJitterGenerator,
    AudioNoiseGenerator,
    CameraNoiseLavaRand,
)

# quantum (optional — requires randeval[quantum])
try:
    from .quantum import (
        QiskitSimulator,
        IBMQuantumBackend,
        MultiQubitHadamard,
        EntanglementBasedQRNG,
        QuantumPhaseEstimationQRNG,
        RandomRotationQRNG,
    )

    _QUANTUM_AVAILABLE = True
except ImportError:
    _QUANTUM_AVAILABLE = False


def list_all() -> list[str]:
    """Return names of all available generator classes.

    Returns:
        list[str]: Sorted list of generator class name strings.
    """
    return list(all_generators().keys())


def all_generators() -> dict[str, type]:
    """Return a dict mapping names to generator classes.

    Returns:
        dict[str, type]: Mapping of {name: class} for every available generator.
    """
    gens: dict[str, type] = {
        "LCG": LCG,
        "MersenneTwister": MersenneTwister,
        "PCG64": PCG64,
        "Philox": Philox,
        "SFC64": SFC64,
        "Xorshift128Plus": Xorshift128Plus,
        "Xoshiro256StarStar": Xoshiro256StarStar,
        "LFSR": LFSR,
        "MiddleSquare": MiddleSquare,
        "MiddleSquareWeyl": MiddleSquareWeyl,
        "WichmannHill": WichmannHill,
        "SystemRandom": SystemRandom,
        "ChaCha20": ChaCha20,
        "AESCTR_DRBG": AESCTR_DRBG,
        "HMAC_DRBG": HMAC_DRBG,
        "Hash_DRBG": Hash_DRBG,
        "BlumBlumShub": BlumBlumShub,
        "RandomOrgGenerator": RandomOrgGenerator,
        "ANUQRNGGenerator": ANUQRNGGenerator,
        "HotBitsGenerator": HotBitsGenerator,
        "RDRANDGenerator": RDRANDGenerator,
        "RDSEEDGenerator": RDSEEDGenerator,
        "HWRNGDevice": HWRNGDevice,
        "TimingJitterGenerator": TimingJitterGenerator,
        "AudioNoiseGenerator": AudioNoiseGenerator,
        "CameraNoiseLavaRand": CameraNoiseLavaRand,
    }
    if _QUANTUM_AVAILABLE:
        gens.update({
            "QiskitSimulator": QiskitSimulator,
            "IBMQuantumBackend": IBMQuantumBackend,
            "MultiQubitHadamard": MultiQubitHadamard,
            "EntanglementBasedQRNG": EntanglementBasedQRNG,
            "QuantumPhaseEstimationQRNG": QuantumPhaseEstimationQRNG,
            "RandomRotationQRNG": RandomRotationQRNG,
        })
    return gens


__all__: list[str] = [
    "Generator",
    "list_all",
    "all_generators",
    # classical
    "LCG", "MersenneTwister", "PCG64", "Philox", "SFC64",
    "Xorshift128Plus", "Xoshiro256StarStar", "LFSR",
    "MiddleSquare", "MiddleSquareWeyl", "WichmannHill",
    # csprng
    "SystemRandom", "ChaCha20", "AESCTR_DRBG", "HMAC_DRBG",
    "Hash_DRBG", "BlumBlumShub",
    # trng
    "RandomOrgGenerator", "ANUQRNGGenerator", "HotBitsGenerator",
    "RDRANDGenerator", "RDSEEDGenerator", "HWRNGDevice",
    "TimingJitterGenerator", "AudioNoiseGenerator", "CameraNoiseLavaRand",
    # quantum (conditional)
    "QiskitSimulator", "IBMQuantumBackend", "MultiQubitHadamard",
    "EntanglementBasedQRNG", "QuantumPhaseEstimationQRNG",
    "RandomRotationQRNG",
]
