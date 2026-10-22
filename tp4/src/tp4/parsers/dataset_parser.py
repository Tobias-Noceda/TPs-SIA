from typing import List

from tp4.types.samples import Samples

import numpy as np
import numpy.typing as npt

import pandas as pd

def parse_dataset(path: str, input_columns: List[str], label_column: str | None = None) -> tuple[Samples, npt.NDArray[np.str_] | None]:
    """
    Parse a numeric CSV into an input matrix X (rows x len(input_columns)), selecting columns by name,
    and the label of each row (None without a label column).
    """
    if not path.endswith(".csv"):
        raise ValueError(f"Dataset path must be a CSV file: {path}")
    ## Check if file exists
    try:
        with open(path, "r") as f:
            pass
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)

    columns = [*input_columns, *([label_column] if label_column else [])]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns not found in {path}: {missing}. Available: {df.columns.tolist()}"
        )

    X = df[input_columns].to_numpy(dtype=np.float64)
    labels = df[label_column].to_numpy(dtype=str) if label_column else None

    return X, labels
