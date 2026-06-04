from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Generator(ABC):
    """Abstract base for all random number generators.

    Generators are stateful — calling generate(n) advances the internal
    state. Same seed on a fresh instance gives the same sequence.
    """

    # True only for quantum generators that run on a classical simulator
    # (not real hardware). Overridden by QiskitSimulator, MultiQubitHadamard,
    # etc. Real QRNGs (IBMQuantumBackend) and every classical generator keep
    # the default False.
    is_simulated: bool = False

    @abstractmethod
    def generate(self, n: int) -> NDArray[np.uint8]:
        """Generate n random bits, advancing internal state.

        Args:
            n: Number of random bits to produce.

        Returns:
            NDArray[np.uint8]: Array of 0s and 1s, length exactly n.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state back to the initial seed.

        Returns:
            None
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable generator name.

        Returns:
            str: Descriptive name for this generator instance.
        """
        ...

    def __repr__(self) -> str:
        """Return a readable string representation with the generator name.

        Returns:
            str: String like 'LCG(name='LCG(a=1103515245, bit=16)')'.
        """
        return f"{type(self).__name__}(name={self.name!r})"
