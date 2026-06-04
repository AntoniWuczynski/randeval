from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from .base import Extractor


class VonNeumannExtractor(Extractor):
    """Von Neumann debiasing: pairs (0,1)->0, (1,0)->1, discard (0,0)/(1,1)."""

    @property
    def name(self) -> str:
        return "VonNeumann"

    def extract(self, data: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if len(data) < 2:
            return np.array([], dtype=np.uint8)
        n = len(data) - len(data) % 2
        pairs = data[:n].reshape(-1, 2)
        diff_mask = pairs[:, 0] != pairs[:, 1]
        kept = pairs[diff_mask]
        if len(kept) == 0:
            return np.array([], dtype=np.uint8)
        return cast("NDArray[np.uint8]", kept[:, 0].copy())
