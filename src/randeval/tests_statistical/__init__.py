"""Statistical randomness tests."""

from .base import StatisticalTest, TestResult, TestSuiteResult, Verdict

# NIST SP 800-22
from .nist import (
    FrequencyTest,
    BlockFrequencyTest,
    RunsTest,
    LongestRunOfOnesTest,
    BinaryMatrixRankTest,
    SpectralTest,
    NonOverlappingTemplateTest,
    OverlappingTemplateTest,
    MaurersUniversalTest,
    LinearComplexityTest,
    SerialTest,
    ApproximateEntropyTest,
    CumulativeSumsTest,
    RandomExcursionsTest,
    RandomExcursionsVariantTest,
    nist_battery,
)

# Dieharder / Marsaglia
from .dieharder import (
    BirthdaySpacingsTest,
    OverlappingPermutationsTest,
    ParkingLotTest,
    MinimumDistanceTest,
    ThreeDSpheresTest,
    SqueezeTest,
    OverlappingSumsTest,
    CrapsTest,
    GCDTest,
    GorillaTest,
    CouponCollectorTest,
    GapTest,
    PokerTest,
    CollisionTest,
    BitstreamTest,
    DNATest,
    CountOnesStreamTest,
    CountOnesByteTest,
    dieharder_battery,
)

# NIST SP 800-90B (entropy estimation for non-IID sources)
from .sp800_90b import (
    MostCommonValueTest,
    CollisionEstimateTest,
    MarkovEstimateTest,
    CompressionEstimateTest,
    TupleEstimateTest,
    LongestRepeatedSubstringTest,
    RepetitionCountTest,
    AdaptiveProportionTest,
    MultiMCWTest,
    LagPredictionTest,
    MultiMMCTest,
    LZ78YTest,
    sp800_90b_battery,
)

# Entropy / information-theoretic
from .entropy import (
    ShannonEntropyTest,
    MinEntropyTest,
    RenyiEntropyTest,
    CompressionRatioTest,
    LempelZivComplexityTest,
    ConditionalEntropyTest,
    MutualInformationTest,
    PermutationEntropyTest,
    entropy_battery,
)

# Distribution
from .distribution import (
    ChiSquaredUniformityTest,
    KolmogorovSmirnovTest,
    AndersonDarlingTest,
    WaldWolfowitzRunsTest,
    MannKendallTrendTest,
    TurningPointTest,
    distribution_battery,
)

# Autocorrelation
from .autocorrelation import AutocorrelationTest

# Novel / randeval-original
from .novel import (
    PValueUniformityTest,
    RunningBiasTest,
    BitPatternSpatialTest,
    WeightDistributionTest,
    ClosePairsTest,
    MaxOfTTest,
    SuccessiveDifferenceTest,
    ByteRunsTest,
    novel_battery,
)


def full_battery() -> list[StatisticalTest]:
    """Return every available test with default parameters.

    Returns:
        list[StatisticalTest]: All NIST, Dieharder, SP 800-90B, entropy,
            distribution, autocorrelation, and novel tests combined.
    """
    return (
        nist_battery()
        + dieharder_battery()
        + sp800_90b_battery()
        + entropy_battery()
        + distribution_battery()
        + [AutocorrelationTest()]
        + novel_battery()
    )


_TEST_NAMES = [
    # NIST SP 800-22
    "FrequencyTest", "BlockFrequencyTest", "RunsTest",
    "LongestRunOfOnesTest", "BinaryMatrixRankTest", "SpectralTest",
    "NonOverlappingTemplateTest", "OverlappingTemplateTest",
    "MaurersUniversalTest", "LinearComplexityTest", "SerialTest",
    "ApproximateEntropyTest", "CumulativeSumsTest",
    "RandomExcursionsTest", "RandomExcursionsVariantTest",
    # Dieharder
    "BirthdaySpacingsTest", "OverlappingPermutationsTest",
    "ParkingLotTest", "MinimumDistanceTest", "ThreeDSpheresTest",
    "SqueezeTest", "OverlappingSumsTest", "CrapsTest", "GCDTest",
    "GorillaTest", "CouponCollectorTest", "GapTest", "PokerTest",
    "CollisionTest", "BitstreamTest", "DNATest",
    "CountOnesStreamTest", "CountOnesByteTest",
    # SP 800-90B
    "MostCommonValueTest", "CollisionEstimateTest",
    "MarkovEstimateTest", "CompressionEstimateTest",
    "TupleEstimateTest", "LongestRepeatedSubstringTest",
    "RepetitionCountTest", "AdaptiveProportionTest",
    "MultiMCWTest", "LagPredictionTest", "MultiMMCTest", "LZ78YTest",
    # Entropy
    "ShannonEntropyTest", "MinEntropyTest", "RenyiEntropyTest",
    "CompressionRatioTest", "LempelZivComplexityTest",
    "ConditionalEntropyTest", "MutualInformationTest",
    "PermutationEntropyTest",
    # Distribution
    "ChiSquaredUniformityTest", "KolmogorovSmirnovTest",
    "AndersonDarlingTest", "WaldWolfowitzRunsTest",
    "MannKendallTrendTest", "TurningPointTest",
    # Autocorrelation
    "AutocorrelationTest",
    # Novel / randeval-original
    "PValueUniformityTest", "RunningBiasTest", "BitPatternSpatialTest",
    "WeightDistributionTest", "ClosePairsTest", "MaxOfTTest",
    "SuccessiveDifferenceTest", "ByteRunsTest",
]


def list_all() -> list[str]:
    """Return the class name of every registered statistical test.

    Returns:
        list[str]: Class names like 'FrequencyTest', 'RunsTest', etc.
    """
    return list(_TEST_NAMES)


__all__ = [
    "StatisticalTest", "TestResult", "TestSuiteResult", "Verdict",
    "full_battery", "list_all",
    "nist_battery", "dieharder_battery", "sp800_90b_battery",
    "entropy_battery", "distribution_battery", "novel_battery",
    # NIST SP 800-22
    "FrequencyTest", "BlockFrequencyTest", "RunsTest",
    "LongestRunOfOnesTest", "BinaryMatrixRankTest", "SpectralTest",
    "NonOverlappingTemplateTest", "OverlappingTemplateTest",
    "MaurersUniversalTest", "LinearComplexityTest", "SerialTest",
    "ApproximateEntropyTest", "CumulativeSumsTest",
    "RandomExcursionsTest", "RandomExcursionsVariantTest",
    # Dieharder
    "BirthdaySpacingsTest", "OverlappingPermutationsTest",
    "ParkingLotTest", "MinimumDistanceTest", "ThreeDSpheresTest",
    "SqueezeTest", "OverlappingSumsTest", "CrapsTest", "GCDTest",
    "GorillaTest", "CouponCollectorTest", "GapTest", "PokerTest",
    "CollisionTest", "BitstreamTest", "DNATest",
    "CountOnesStreamTest", "CountOnesByteTest",
    # SP 800-90B
    "MostCommonValueTest", "CollisionEstimateTest",
    "MarkovEstimateTest", "CompressionEstimateTest",
    "TupleEstimateTest", "LongestRepeatedSubstringTest",
    "RepetitionCountTest", "AdaptiveProportionTest",
    "MultiMCWTest", "LagPredictionTest", "MultiMMCTest", "LZ78YTest",
    # Entropy
    "ShannonEntropyTest", "MinEntropyTest", "RenyiEntropyTest",
    "CompressionRatioTest", "LempelZivComplexityTest",
    "ConditionalEntropyTest", "MutualInformationTest",
    "PermutationEntropyTest",
    # Distribution
    "ChiSquaredUniformityTest", "KolmogorovSmirnovTest",
    "AndersonDarlingTest", "WaldWolfowitzRunsTest",
    "MannKendallTrendTest", "TurningPointTest",
    # Autocorrelation
    "AutocorrelationTest",
    # Novel / randeval-original
    "PValueUniformityTest", "RunningBiasTest", "BitPatternSpatialTest",
    "WeightDistributionTest", "ClosePairsTest", "MaxOfTTest",
    "SuccessiveDifferenceTest", "ByteRunsTest",
]
