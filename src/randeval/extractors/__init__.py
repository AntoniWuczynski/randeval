"""Randomness extractors."""

from .base import Extractor
from .von_neumann import VonNeumannExtractor

# debiasing family
from .debiasing import PeresExtractor, EliasExtractor, AMLSExtractor

# universal hashing
from .hashing import (
    ToeplitzHash,
    LinearHash,
    InnerProductExtractor,
    LHLHash,
    PolynomialHash,
)

# computational
from .computational import HMACExtractor, SHAConditioner

# practical / hardware
from .practical import (
    XORExtractor,
    BlockParityExtractor,
    BitDecimationExtractor,
    WindowedXORExtractor,
    RepetitionFilter,
    ModularReductionExtractor,
    CRCExtractor,
    SubsamplingExtractor,
    Condenser,
)

# adaptive
from .adaptive import (
    MinEntropyExtractor,
    FuzzyExtractor,
    ArithmeticCodingExtractor,
)

# cryptomite (optional)
try:
    from .cryptomite_ext import (
        CryptoMiteToeplitz,
        CryptoMiteCirculant,
        CryptoMiteDodis,
        CryptoMiteTrevisan,
    )
    _CRYPTOMITE_AVAILABLE = True
except ImportError:
    _CRYPTOMITE_AVAILABLE = False


def all_extractors() -> dict[str, type]:
    """Return a dict mapping names to extractor classes."""
    exts: dict[str, type] = {
        "VonNeumann": VonNeumannExtractor,
        "Peres": PeresExtractor,
        "Elias": EliasExtractor,
        "AMLS": AMLSExtractor,
        "ToeplitzHash": ToeplitzHash,
        "LinearHash": LinearHash,
        "InnerProduct": InnerProductExtractor,
        "LHLHash": LHLHash,
        "PolynomialHash": PolynomialHash,
        "HMAC": HMACExtractor,
        "SHAConditioner": SHAConditioner,
        "XOR": XORExtractor,
        "BlockParity": BlockParityExtractor,
        "BitDecimation": BitDecimationExtractor,
        "WindowedXOR": WindowedXORExtractor,
        "RepetitionFilter": RepetitionFilter,
        "ModularReduction": ModularReductionExtractor,
        "CRC32": CRCExtractor,
        "Subsampling": SubsamplingExtractor,
        "Condenser": Condenser,
        "MinEntropy": MinEntropyExtractor,
        "Fuzzy": FuzzyExtractor,
        "ArithmeticCoding": ArithmeticCodingExtractor,
    }
    if _CRYPTOMITE_AVAILABLE:
        exts.update({
            "CryptoMite-Toeplitz": CryptoMiteToeplitz,
            "CryptoMite-Circulant": CryptoMiteCirculant,
            "CryptoMite-Dodis": CryptoMiteDodis,
            "CryptoMite-Trevisan": CryptoMiteTrevisan,
        })
    return exts


def list_all() -> list[str]:
    """Return names of all available extractor classes."""
    return list(all_extractors().keys())


def default_extractors() -> list[Extractor]:
    """Return instances of all extractors with default params (no optional deps)."""
    return [
        VonNeumannExtractor(),
        PeresExtractor(),
        EliasExtractor(bias=0.6),
        AMLSExtractor(),
        ToeplitzHash(),
        LinearHash(),
        InnerProductExtractor(),
        LHLHash(),
        PolynomialHash(),
        HMACExtractor(),
        SHAConditioner(),
        XORExtractor(),
        BlockParityExtractor(),
        BitDecimationExtractor(),
        WindowedXORExtractor(),
        RepetitionFilter(),
        ModularReductionExtractor(block_bits=8, modulus=127),
        CRCExtractor(),
        SubsamplingExtractor(),
        Condenser(),
        MinEntropyExtractor(),
        FuzzyExtractor(),
        ArithmeticCodingExtractor(bias=0.6),
    ]


__all__: list[str] = [
    "Extractor",
    "list_all",
    "all_extractors",
    "default_extractors",
    # debiasing
    "VonNeumannExtractor", "PeresExtractor", "EliasExtractor", "AMLSExtractor",
    # hashing
    "ToeplitzHash", "LinearHash", "InnerProductExtractor", "LHLHash", "PolynomialHash",
    # computational
    "HMACExtractor", "SHAConditioner",
    # practical
    "XORExtractor", "BlockParityExtractor", "BitDecimationExtractor",
    "WindowedXORExtractor", "RepetitionFilter", "ModularReductionExtractor",
    "CRCExtractor", "SubsamplingExtractor", "Condenser",
    # adaptive
    "MinEntropyExtractor", "FuzzyExtractor", "ArithmeticCodingExtractor",
    # cryptomite (conditional)
    "CryptoMiteToeplitz", "CryptoMiteCirculant", "CryptoMiteDodis", "CryptoMiteTrevisan",
]
