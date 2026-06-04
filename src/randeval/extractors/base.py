from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Extractor(ABC):
    """Abstract base for randomness extractors."""

    @abstractmethod
    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Apply extraction to a bit array, returning a (usually shorter) bit array.

        Args:
            data: 1-D numpy array of 0s and 1s.

        Returns:
            NDArray[np.uint8]: Extracted bit array.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this extractor.

        Returns:
            str: Display name, e.g. "VonNeumann".
        """
        ...
