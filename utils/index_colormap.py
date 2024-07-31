from __future__ import annotations

from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

DI = TypeVar("DI", np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64)


class IndexColorMap:

    _index_colormap: NDArray[np.uint8]

    def __init__(self, n: int = 10000):
        if n <= 0:
            raise ValueError(f"\"n\" must be more than 0, but got \"{n}\"")

        self._index_colormap = self._generate_color_map(n)

    # ------------------------------------------------------------------------------------------------------------------
    #
    #   Public Method
    #
    # ------------------------------------------------------------------------------------------------------------------
    def colorize(self, indices: (int | NDArray[DI]), max_index: int = -1) -> NDArray[np.uint8]:
        """
        Args:
            indices: [...] shape.
            max_index:

        Returns:
            [..., 3] shape.
        """
        if (max_index != -1) and (len(self._index_colormap) <= max_index):
            self._index_colormap = self._generate_color_map(max_index + 1)
        return self._index_colormap[indices]
    
    def get_color(self, id: int) -> NDArray[np.float32]:
        return self._index_colormap[id] / 255

    # ------------------------------------------------------------------------------------------------------------------
    #
    #   Private Method
    #
    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _generate_color_map(n: int) -> NDArray[np.uint8]:
        """
        Args:
            n:

        Returns:
            [N, 3]
        """
        if 256**3 < n:
            raise ValueError(f"n must be less than 16777215 (256^3), but got {n}")

        k = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
        shift = np.arange(n)[..., None] >> (k * 3)
        bit = (shift[..., None] & np.array([1, 2, 4])) != 0
        rgb_or = bit << 7 - k[None, :, None]
        rgb = (
            rgb_or[:, 0]
            | rgb_or[:, 1]
            | rgb_or[:, 2]
            | rgb_or[:, 3]
            | rgb_or[:, 4]
            | rgb_or[:, 5]
            | rgb_or[:, 6]
            | rgb_or[:, 7]
        )

        return rgb
