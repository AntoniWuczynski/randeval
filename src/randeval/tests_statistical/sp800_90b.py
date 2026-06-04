"""NIST SP 800-90B entropy estimation tests.

These estimate the min-entropy of non-IID sources — exactly what raw
TRNG and QRNG output is. Distinct from SP 800-22 which tests
statistical quality of (assumed IID) sequences.

Reference: NIST SP 800-90B, Recommendation for the Entropy Sources
Used for Random Bit Generation (January 2018).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray

from .base import StatisticalTest, TestResult, Verdict
from ._utils import bits_to_blocks, verdict


def _entropy_result(name: str, h_min: float, block_size: int) -> TestResult:
    """Build a TestResult for an entropy estimator.

    Args:
        name: Test name for the result.
        h_min: Estimated min-entropy in bits.
        block_size: Block size in bits (used to normalise and set pass threshold).

    Returns:
        TestResult: Statistic is h_min, p_value is normalised entropy, verdict
            passes if h_min >= 0.3 * block_size.
    """
    h_min = max(0.0, min(h_min, float(block_size)))
    normalized = h_min / block_size if block_size > 0 else 0.0
    v = Verdict.PASS if h_min >= 0.3 * block_size else Verdict.FAIL
    return TestResult(test_name=name, statistic=h_min, p_value=normalized, verdict=v)


# ── Section 6.3: IID Track Estimators ────────────────────────


class MostCommonValueTest(StatisticalTest):
    """Most Common Value Estimate (SP 800-90B §6.3.1).

    Estimates min-entropy from the frequency of the most common
    sample value. H = -log2(p_max). Simplest estimator — baseline
    for all others.
    """

    def __init__(self, *, block_size: int = 8) -> None:
        """Configure block size in bits for sample grouping.

        Args:
            block_size: Number of bits per sample (b).
        """
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'SP800-90B: Most Common Value (b=8)'.
        """
        return f"SP800-90B: Most Common Value (b={self._block_size})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Estimate min-entropy from the frequency of the most common sample value.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Min-entropy estimate as statistic, normalised p-value, and verdict.
        """
        samples = bits_to_blocks(data, self._block_size)
        n = len(samples)
        if n == 0:
            return _entropy_result(self.name, 0.0, self._block_size)

        _, counts = np.unique(samples, return_counts=True)
        p_max = float(counts.max()) / n

        # upper bound with z=2.576 (99% confidence)
        p_u = min(1.0, p_max + 2.576 * math.sqrt(p_max * (1 - p_max) / n))
        h_min = -math.log2(p_u) if p_u > 0 else float(self._block_size)

        return _entropy_result(self.name, h_min, self._block_size)


class CollisionEstimateTest(StatisticalTest):
    """Collision Estimate (SP 800-90B §6.3.2).

    Measures average distance between repeated sample values.
    Shorter distances -> lower entropy. Uses the mean collision
    time to bound min-entropy.
    """

    def __init__(self, *, block_size: int = 8) -> None:
        """Configure block size in bits for sample grouping.

        Args:
            block_size: Number of bits per sample (b).
        """
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'SP800-90B: Collision Estimate (b=8)'.
        """
        return f"SP800-90B: Collision Estimate (b={self._block_size})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Measure mean collision distance and estimate min-entropy from it.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Min-entropy estimate as statistic, normalised p-value, and verdict.
        """
        samples = bits_to_blocks(data, self._block_size)
        n = len(samples)
        if n < 3:
            return _entropy_result(self.name, 0.0, self._block_size)

        # measure collision distances per SP 800-90B: scan forward from i,
        # record distance when any previously-seen value reappears, then restart
        distances: list[int] = []
        i = 0
        while i < n:
            seen = {int(samples[i])}
            j = i + 1
            while j < n:
                v = int(samples[j])
                if v in seen:
                    distances.append(j - i)
                    break
                seen.add(v)
                j += 1
            else:
                break
            i = j + 1

        if not distances:
            # no collisions at all — max entropy
            return _entropy_result(self.name, float(self._block_size), self._block_size)

        mean_d = float(np.mean(distances))
        # H ≈ log2(mean_distance) - 0.5
        h_min = max(0.0, math.log2(mean_d) - 0.5) if mean_d > 1 else 0.0

        return _entropy_result(self.name, h_min, self._block_size)


class MarkovEstimateTest(StatisticalTest):
    """Markov Estimate (SP 800-90B §6.3.3).

    Fits a first-order Markov model to the sample sequence and
    computes the per-symbol entropy of the most likely path.
    Captures sequential dependence that IID estimators miss.
    """

    def __init__(self) -> None:
        """No configuration needed -- operates on raw bits."""
        pass

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'SP800-90B: Markov Estimate'.
        """
        return "SP800-90B: Markov Estimate"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Fit a first-order Markov model and estimate per-symbol min-entropy.

        Args:
            data: 1-D array of 0s and 1s (raw bits, not blocks).

        Returns:
            TestResult: Min-entropy estimate as statistic, normalised p-value, and verdict.
        """
        # markov estimate works on raw bits, not blocks
        bits = data.astype(np.float64)
        n = len(bits)
        if n < 256:
            return _entropy_result(self.name, 0.0, 1)

        # count transitions (vectorised)
        pairs = data[:-1].astype(np.int32) * 2 + data[1:].astype(np.int32)
        tc = np.bincount(pairs, minlength=4)
        t00, t01, t10, t11 = int(tc[0]), int(tc[1]), int(tc[2]), int(tc[3])

        # transition probabilities
        n0 = t00 + t01
        n1 = t10 + t11
        if n0 == 0 or n1 == 0:
            return _entropy_result(self.name, 0.0, 1)

        p00 = t00 / n0
        p01 = t01 / n0
        p10 = t10 / n1
        p11 = t11 / n1

        # initial state probs
        p_0 = n0 / (n - 1)
        p_1 = n1 / (n - 1)

        path_len = 128
        h_min = float('inf')
        for start in (0, 1):
            state = start
            init_p = p_0 if state == 0 else p_1
            log_prob = math.log2(init_p) if init_p > 0 else -1e10
            for _ in range(path_len):
                if state == 0:
                    tp = max(p00, p01)
                    state = 0 if p00 >= p01 else 1
                else:
                    tp = max(p10, p11)
                    state = 1 if p11 >= p10 else 0
                log_prob += math.log2(tp) if tp > 0 else -1e10
            h = -log_prob / path_len
            h_min = min(h_min, h)

        return _entropy_result(self.name, h_min, 1)


class CompressionEstimateTest(StatisticalTest):
    """Compression Estimate (SP 800-90B §6.3.4).

    Uses Maurer's universal test statistic to estimate entropy
    from compressibility. Related to MaurersUniversalTest in the
    NIST 800-22 suite but tuned for entropy estimation.
    """

    def __init__(self, *, block_length: int = 6) -> None:
        """Configure the L-bit block length for Maurer-style compression.

        Args:
            block_length: Number of bits per block (L).
        """
        self._block_length = block_length

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'SP800-90B: Compression Estimate (L=6)'.
        """
        return f"SP800-90B: Compression Estimate (L={self._block_length})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Estimate entropy from mean log-distance between repeated patterns.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Min-entropy estimate as statistic, normalised p-value, and verdict.
        """
        L = self._block_length
        samples = bits_to_blocks(data, L)
        n = len(samples)
        Q = 2 ** L  # init segment length
        if n < Q + 1:
            return _entropy_result(self.name, 0.0, L)

        # init: record last occurrence of each block in first Q samples
        last_seen: dict[int, int] = {}
        for i in range(Q):
            last_seen[int(samples[i])] = i

        # test segment
        dists: list[float] = []
        for i in range(Q, n):
            v = int(samples[i])
            if v in last_seen:
                d = i - last_seen[v]
                dists.append(math.log2(d))
            last_seen[v] = i

        if not dists:
            return _entropy_result(self.name, float(L), L)

        mean_log = float(np.mean(dists))
        h_min = max(0.0, mean_log)

        return _entropy_result(self.name, h_min, L)


class TupleEstimateTest(StatisticalTest):
    """t-Tuple Estimate (SP 800-90B §6.3.5).

    Counts frequencies of t-tuples and (t+1)-tuples to estimate
    the collision probability. The maximum over tuple lengths
    gives the entropy bound.
    """

    def __init__(self, *, block_size: int = 8) -> None:
        """Configure block size in bits for sample grouping.

        Args:
            block_size: Number of bits per sample (b).
        """
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'SP800-90B: t-Tuple Estimate (b=8)'.
        """
        return f"SP800-90B: t-Tuple Estimate (b={self._block_size})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Find the most frequent t-tuple across increasing t and bound min-entropy.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Min-entropy estimate as statistic, normalised p-value, and verdict.
        """
        samples = bits_to_blocks(data, self._block_size)
        n = len(samples)
        if n < 2:
            return _entropy_result(self.name, 0.0, self._block_size)

        # for t=1,2,3,... find max frequency of any t-tuple
        h_estimates: list[float] = []
        for t in range(1, min(n, 20) + 1):
            counts: dict[tuple[int, ...], int] = defaultdict(int)
            for i in range(n - t + 1):
                key = tuple(int(x) for x in samples[i:i + t])
                counts[key] += 1

            if not counts:
                break
            max_freq = max(counts.values())
            total = n - t + 1
            p_max = max_freq / total
            if p_max <= 0:
                break

            # per-sample entropy from t-tuple
            h = -math.log2(p_max) / t
            h_estimates.append(h)

            # stop when no tuple repeats
            if max_freq <= 1:
                break

        h_min = min(h_estimates) if h_estimates else 0.0
        return _entropy_result(self.name, h_min, self._block_size)


class LongestRepeatedSubstringTest(StatisticalTest):
    """Longest Repeated Substring Estimate (SP 800-90B §6.3.6).

    Finds the longest substring that appears at least twice.
    Longer repeated substrings indicate lower entropy. Uses a
    suffix-array based approach for efficiency.
    """

    def __init__(self, *, block_size: int = 8) -> None:
        """Configure block size in bits for sample grouping.

        Args:
            block_size: Number of bits per sample (b).
        """
        self._block_size = block_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'SP800-90B: LRS Estimate (b=8)'.
        """
        return f"SP800-90B: LRS Estimate (b={self._block_size})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Binary-search for the longest repeated substring and estimate entropy from it.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Min-entropy estimate as statistic, normalised p-value, and verdict.
        """
        samples = bits_to_blocks(data, self._block_size)
        n = len(samples)
        if n < 10:
            return _entropy_result(self.name, 0.0, self._block_size)

        # find longest substring appearing at least twice, O(n^2)-ish
        # cap search length to keep it tractable
        max_len = min(n // 2, 500)
        longest = 0

        # use a set-based approach: for each length, hash all substrings
        lo, hi = 1, max_len
        while lo <= hi:
            mid = (lo + hi) // 2
            seen: set[tuple[int, ...]] = set()
            found = False
            for i in range(n - mid + 1):
                t = tuple(int(x) for x in samples[i:i + mid])
                if t in seen:
                    found = True
                    break
                seen.add(t)
            if found:
                longest = mid
                lo = mid + 1
            else:
                hi = mid - 1

        # H ≈ log2(n) / (longest + 1)
        h_min = math.log2(n) / (longest + 1) if longest >= 0 else float(self._block_size)

        return _entropy_result(self.name, h_min, self._block_size)


# ── Section 5: Health Tests ──────────────────────────────────


class RepetitionCountTest(StatisticalTest):
    """Repetition Count Test (SP 800-90B §5.1).

    Online health test. Detects when the source produces too many
    identical consecutive outputs. Flags catastrophic failure
    (stuck source). Cutoff C depends on assumed min-entropy H.
    """

    def __init__(self, *, assumed_entropy: float = 1.0) -> None:
        """Configure assumed min-entropy H used to compute the cutoff threshold.

        Args:
            assumed_entropy: Assumed per-bit min-entropy H (determines max-run cutoff).
        """
        self._assumed_entropy = assumed_entropy

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'SP800-90B: Repetition Count (H=1.0)'.
        """
        return f"SP800-90B: Repetition Count (H={self._assumed_entropy})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Scan for the longest run of identical consecutive values and check against cutoff.

        Args:
            data: 1-D array of 0s and 1s (or raw sample bytes).

        Returns:
            TestResult: Max run length as statistic, normalised p-value, and verdict.
        """
        samples = data  # works on raw bytes/bits
        n = len(samples)
        H = self._assumed_entropy
        alpha = 2.0 ** (-20)  # standard false positive rate

        # cutoff: longest allowable run, scaled for sequence length
        # SP 800-90B formula is for online monitoring; for post-hoc analysis
        # of n bits, the expected longest run is ~log2(n), so we add that
        base_C = math.ceil(-math.log2(alpha) / H) if H > 0 else n
        C = base_C + math.ceil(math.log2(max(n, 2)))

        # scan for runs of identical values
        max_run = 1
        cur_run = 1
        for i in range(1, n):
            if samples[i] == samples[i - 1]:
                cur_run += 1
                max_run = max(max_run, cur_run)
            else:
                cur_run = 1

        failed = max_run >= C
        # statistic is max run length; p_value: normalized run vs cutoff
        p_val = max(0.0, 1.0 - max_run / C) if C > 0 else 0.0
        v = Verdict.FAIL if failed else Verdict.PASS

        return TestResult(
            test_name=self.name,
            statistic=float(max_run),
            p_value=p_val,
            verdict=v,
        )


class AdaptiveProportionTest(StatisticalTest):
    """Adaptive Proportion Test (SP 800-90B §5.2).

    Online health test. Tracks the proportion of a specific value
    within a sliding window. Detects bias drift — e.g. a QRNG
    whose laser is degrading over time.
    """

    def __init__(self, *, window_size: int = 512) -> None:
        """Configure the sliding window size W for proportion tracking.

        Args:
            window_size: Number of samples per window (W).
        """
        self._window_size = window_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'SP800-90B: Adaptive Proportion (W=512)'.
        """
        return f"SP800-90B: Adaptive Proportion (W={self._window_size})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Slide a window and check if any value's proportion exceeds the binomial cutoff.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Max window proportion as statistic, p-value, and verdict.
        """
        n = len(data)
        W = self._window_size
        K = n // W
        if K < 1:
            return TestResult(
                test_name=self.name, statistic=0.0,
                p_value=1.0, verdict=Verdict.PASS,
            )

        from scipy.stats import binom
        # non-overlapping windows; the 2^-20 cutoff in the spec is a per-window
        # false-alarm budget, so sliding it over n-W+1 windows guaranteed a fail
        # on any long good stream. Test the single most extreme window and
        # Sidak-correct for the K windows actually examined.
        blocks = data[:K * W].reshape(K, W)
        ones = blocks.sum(axis=1)
        majority = np.maximum(ones, W - ones)
        cmax = int(majority.max())

        # two-sided binomial tail for the most extreme window
        p_window = min(1.0, 2.0 * float(binom.sf(cmax - 1, W, 0.5)))
        p_overall = 1.0 - (1.0 - p_window) ** K

        return TestResult(
            test_name=self.name,
            statistic=cmax / W,
            p_value=p_overall,
            verdict=verdict(p_overall),
        )


# ── Section 6.3: Prediction Estimators ───────────────────────


class MultiMCWTest(StatisticalTest):
    """Multi Most Common in Window Estimate (SP 800-90B §6.3.7).

    Prediction estimator. Uses the most common value within a
    sliding window to predict the next output. Higher prediction
    accuracy → lower entropy.
    """

    def __init__(self, *, window_sizes: list[int] | None = None) -> None:
        """Configure the set of window sizes to try for most-common-value prediction.

        Args:
            window_sizes: List of window widths to evaluate (default [63, 255, 1023, 4095]).
        """
        self._window_sizes = window_sizes or [63, 255, 1023, 4095]

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'SP800-90B: MultiMCW Predictor'.
        """
        return "SP800-90B: MultiMCW Predictor"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Predict next value as the most common in a sliding window; estimate entropy from accuracy.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Min-entropy estimate as statistic, normalised p-value, and verdict.
        """
        samples = bits_to_blocks(data, 8)  # work on byte-level samples
        n = len(samples)
        bs = 8

        if n < 100:
            return _entropy_result(self.name, 0.0, bs)

        best_acc = 0.0
        for w in self._window_sizes:
            if w >= n:
                continue
            correct = 0
            total = 0
            for i in range(w, n):
                # predict as most common in window
                window = samples[i - w:i]
                counts: dict[int, int] = defaultdict(int)
                for v in window:
                    counts[int(v)] += 1
                prediction = max(counts, key=counts.get)  # type: ignore[arg-type]
                if prediction == int(samples[i]):
                    correct += 1
                total += 1

            if total > 0:
                acc = correct / total
                best_acc = max(best_acc, acc)

        if best_acc <= 0 or best_acc >= 1.0:
            h_min = 0.0 if best_acc >= 1.0 else float(bs)
        else:
            h_min = -math.log2(best_acc)

        return _entropy_result(self.name, h_min, bs)


class LagPredictionTest(StatisticalTest):
    """Lag Prediction Estimate (SP 800-90B §6.3.8).

    Predicts the next output as the value seen D steps ago.
    Tests multiple lag values and uses the best predictor.
    """

    def __init__(self, *, max_lag: int = 128) -> None:
        """Configure maximum lag D to test for prediction.

        Args:
            max_lag: Largest lag value to try.
        """
        self._max_lag = max_lag

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'SP800-90B: Lag Predictor (D<=128)'.
        """
        return f"SP800-90B: Lag Predictor (D≤{self._max_lag})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Predict next value as the value D steps ago; estimate entropy from best accuracy.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Min-entropy estimate as statistic, normalised p-value, and verdict.
        """
        samples = bits_to_blocks(data, 8)
        n = len(samples)
        bs = 8

        if n < self._max_lag + 10:
            return _entropy_result(self.name, 0.0, bs)

        best_acc = 0.0
        for d in range(1, min(self._max_lag, n) + 1):
            correct = 0
            total = n - d
            if total <= 0:
                break
            # vectorized comparison
            matches = (samples[d:] == samples[:-d])
            acc = float(np.sum(matches)) / total
            best_acc = max(best_acc, acc)

        if best_acc <= 0:
            h_min = float(bs)
        elif best_acc >= 1.0:
            h_min = 0.0
        else:
            h_min = -math.log2(best_acc)

        return _entropy_result(self.name, h_min, bs)


class MultiMMCTest(StatisticalTest):
    """Multi Markov Model with Counting Estimate (SP 800-90B §6.3.9).

    Prediction estimator using Markov models of varying orders.
    Predicts next output from the most recently seen context.
    Tests orders 1 through D.
    """

    def __init__(self, *, max_order: int = 16) -> None:
        """Configure maximum Markov model order to test.

        Args:
            max_order: Highest Markov order D to evaluate.
        """
        self._max_order = max_order

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Like 'SP800-90B: MultiMMC Predictor (D<=16)'.
        """
        return f"SP800-90B: MultiMMC Predictor (D≤{self._max_order})"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Build Markov models of orders 1..D, predict from context, estimate entropy.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Min-entropy estimate as statistic, normalised p-value, and verdict.
        """
        samples = bits_to_blocks(data, 8)
        n = len(samples)
        bs = 8

        if n < 50:
            return _entropy_result(self.name, 0.0, bs)

        best_acc = 0.0
        for order in range(1, min(self._max_order, n) + 1):
            # markov model of given order: context -> {value: count}
            model: dict[tuple[int, ...], dict[int, int]] = defaultdict(lambda: defaultdict(int))
            correct = 0
            total = 0

            for i in range(order, n):
                ctx = tuple(int(x) for x in samples[i - order:i])
                actual = int(samples[i])

                # predict from model if we have context
                if ctx in model and model[ctx]:
                    prediction = max(model[ctx], key=model[ctx].get)  # type: ignore[arg-type]
                    if prediction == actual:
                        correct += 1

                model[ctx][actual] += 1
                total += 1

            if total > 0:
                acc = correct / total
                best_acc = max(best_acc, acc)

        if best_acc <= 0:
            h_min = float(bs)
        elif best_acc >= 1.0:
            h_min = 0.0
        else:
            h_min = -math.log2(best_acc)

        return _entropy_result(self.name, h_min, bs)


class LZ78YTest(StatisticalTest):
    """LZ78Y Prediction Estimate (SP 800-90B §6.3.10).

    Prediction estimator based on the LZ78 parsing dictionary.
    Builds a dictionary of observed patterns and predicts from
    the longest matching context. Good at capturing complex
    sequential structure.
    """

    def __init__(self, *, max_dict_size: int = 65536) -> None:
        """Configure the maximum dictionary size before context entries are dropped.

        Args:
            max_dict_size: Cap on the number of dictionary entries.
        """
        self._max_dict_size = max_dict_size

    @property
    def name(self) -> str:
        """Human-readable test name.

        Returns:
            str: Always 'SP800-90B: LZ78Y Predictor'.
        """
        return "SP800-90B: LZ78Y Predictor"

    def run(self, data: NDArray[np.uint8]) -> TestResult:
        """Predict from the longest matching LZ78 context and estimate entropy from accuracy.

        Args:
            data: 1-D array of 0s and 1s.

        Returns:
            TestResult: Min-entropy estimate as statistic, normalised p-value, and verdict.
        """
        samples = bits_to_blocks(data, 8)
        n = len(samples)
        bs = 8

        if n < 50:
            return _entropy_result(self.name, 0.0, bs)

        # LZ78-style dictionary predictor
        # dict maps context tuple -> {next_value: count}
        dictionary: dict[tuple[int, ...], dict[int, int]] = defaultdict(lambda: defaultdict(int))
        max_ctx = 16  # max context length to look back

        correct = 0
        total = 0

        for i in range(1, n):
            actual = int(samples[i])
            prediction = None
            # try longest matching context first
            for length in range(min(i, max_ctx), 0, -1):
                ctx = tuple(int(x) for x in samples[i - length:i])
                if ctx in dictionary and dictionary[ctx]:
                    prediction = max(dictionary[ctx], key=dictionary[ctx].get)  # type: ignore[arg-type]
                    break

            if prediction is not None:
                if prediction == actual:
                    correct += 1
                total += 1

            # update dictionary with all sub-contexts
            for length in range(1, min(i, max_ctx) + 1):
                ctx = tuple(int(x) for x in samples[i - length:i])
                dictionary[ctx][actual] += 1
                if len(dictionary) > self._max_dict_size:
                    break

        if total <= 0:
            h_min = float(bs)
        else:
            acc = correct / total
            if acc <= 0:
                h_min = float(bs)
            elif acc >= 1.0:
                h_min = 0.0
            else:
                h_min = -math.log2(acc)

        return _entropy_result(self.name, h_min, bs)


# ── Suite runner ─────────────────────────────────────────────

def sp800_90b_battery() -> list[StatisticalTest]:
    """Return all SP 800-90B entropy estimation tests with defaults.

    Returns:
        list[StatisticalTest]: 12 test instances covering IID estimators,
            health tests, and prediction estimators.
    """
    return [
        # IID estimators
        MostCommonValueTest(),
        CollisionEstimateTest(),
        MarkovEstimateTest(),
        CompressionEstimateTest(),
        TupleEstimateTest(),
        LongestRepeatedSubstringTest(),
        # Health tests
        RepetitionCountTest(),
        AdaptiveProportionTest(),
        # Prediction estimators
        MultiMCWTest(),
        LagPredictionTest(),
        MultiMMCTest(),
        LZ78YTest(),
    ]
