from typing import Tuple

from tp5.types.float import Float

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

def draw_glyph(
    ax: Axes,
    pattern: npt.NDArray[Float],
    shape: Tuple[int, int],
    wrong: npt.NDArray[np.bool_] | None = None,
    title: str | None = None,
    title_color: str = "black",
) -> None:
    """
    Draw one pattern as a grayscale glyph (1 = black). `wrong` (same length as the pattern) outlines in
    red the pixels that differ from the original.
    """
    ax.imshow(np.asarray(pattern).reshape(shape), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    if wrong is not None:
        for index in np.flatnonzero(wrong):
            row, column = divmod(int(index), shape[1])
            ax.add_patch(Rectangle((column - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.5))
    if title is not None:
        ax.set_title(title, fontsize=8, color=title_color)
    ax.set_xticks([])
    ax.set_yticks([])
