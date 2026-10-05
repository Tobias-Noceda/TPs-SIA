from typing import List

from tp3.types.float import Float

import numpy as np
import numpy.typing as npt

import pandas as pd

import ast

def parse_dataset(
    path: str, input_columns: List[str] | str, expected_output: List[str]
) -> tuple[npt.NDArray[Float], npt.NDArray[Float]]:
    """
    Parse a numeric CSV into an input matrix X (rows x len(input_columns))
    and an expected output vector y, selecting columns by name.
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

    input_is_array = isinstance(input_columns, str)
    columns = [input_columns] if input_is_array else input_columns

    missing = [c for c in [*columns, *expected_output] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns not found in {path}: {missing}. Available: {df.columns.tolist()}"
        )

    if input_is_array:
        df[input_columns] = df[input_columns].map(_parse_array)
        X = df[input_columns].to_numpy(dtype=np.ndarray)
    else:
        X = df[columns].to_numpy(dtype=np.float64)

    y = df[expected_output].to_numpy(dtype=np.float64)

    return X, y

def _parse_array(d: object) -> npt.NDArray[Float]:
    return np.array(ast.literal_eval(str(d)), dtype=np.float64)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Parse a CSV dataset into X, y.")
    # parser.add_argument("path", help="Path to the CSV dataset")
    # parser.add_argument(
    #     "-i", "--inputs", nargs="+", required=True, help="Input column names"
    # )
    # parser.add_argument(
    #     "-o", "--output", required=True, help="Expected output column name"
    # )
    # args = parser.parse_args()

    # X, y = parse_dataset(args.path, args.inputs, args.output)

    X, y = parse_dataset(
        "fraud_dataset.csv",
        input_columns=["timestamp", "amount_usd", "quantity_purchased", "session_duration_seconds", "days_since_last_purchase", "account_age_days", "device_screen_resolution", "time_since_last_login_s", "items_viewed_before_purchase"],
        expected_output=["big_model_fraud_probability"],
    )

