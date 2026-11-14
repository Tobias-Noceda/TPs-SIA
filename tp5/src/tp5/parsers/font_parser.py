import re
from typing import List, Tuple

from tp5.types.samples import Samples

import numpy as np

# Rows of the C array: `{0x04, 0x04, ...},   // 0x60, \``
_ROW = re.compile(r"\{((?:\s*0x[0-9a-fA-F]+\s*,?)+)\}\s*,?\s*//\s*0x[0-9a-fA-F]+\s*,\s*(\S+)")

def parse_font(path: str, width: int = 5) -> Tuple[Samples, List[str], Tuple[int, int]]:
    """
    Parse the glyphs of a `font.h` file.

    Each row of the glyph is a byte whose `width` lowest bits are the pixels, the most significant
    one being the leftmost column.

    Returns:
        Patterns (glyphs, rows · width) in {0, 1} flattened row-major, their labels (the character in
        the comment) and the glyph shape (rows, width).
    """
    with open(path, "r") as file:
        content = file.read()

    patterns = []
    labels = []
    for values, label in _ROW.findall(content):
        rows = [int(value, 16) for value in values.replace(",", " ").split()]
        bits = [(row >> (width - 1 - column)) & 1 for row in rows for column in range(width)]
        patterns.append(bits)
        labels.append(label)

    if not patterns:
        raise ValueError(f"No glyphs found in '{path}'.")

    heights = {len(p) // width for p in patterns}
    if len(heights) != 1:
        raise ValueError(f"Glyphs of '{path}' have different heights: {sorted(heights)}.")

    return np.array(patterns, dtype=np.float64), labels, (heights.pop(), width)
