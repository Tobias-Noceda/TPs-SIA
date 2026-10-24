from pathlib import Path

from tp4.types.pattern import Pattern

import numpy as np
import matplotlib.pyplot as plt

def parse_pattern(path: str) -> tuple[Pattern, tuple[int, int]]:
    """
    Parse a binary image into a ±1 pattern (N,) and its (rows, columns) shape, row by row.

    `.png`: dark pixels (gray < 0.5) are 1, light ones -1. `.txt`: one row per line, `#` is 1 and `.` is -1.
    """
    file = Path(path)
    if not file.is_file():
        raise ValueError(f"Pattern file not found: {path}")

    if file.suffix == ".png":
        image = plt.imread(file)
        # RGB(A) images become gray by averaging the color channels
        if image.ndim == 3:
            image = image[..., :3].mean(axis=2)
        grid = np.where(image < 0.5, 1.0, -1.0)
    elif file.suffix == ".txt":
        rows = [line.strip() for line in file.read_text().splitlines() if line.strip()]
        if len({len(row) for row in rows}) != 1 or set("".join(rows)) - {"#", "."}:
            raise ValueError(f"Pattern {path} must be a rectangle of '#' and '.'.")
        grid = np.array([[1.0 if c == "#" else -1.0 for c in row] for row in rows])
    else:
        raise ValueError(f"Pattern file must be .png or .txt: {path}")

    return grid.ravel(), (grid.shape[0], grid.shape[1])
