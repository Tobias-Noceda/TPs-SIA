import argparse

import numpy as np
import pandas as pd


def parse_dataset(
    path: str, input_columns: list[str], expected_output: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a numeric CSV into an input matrix X (rows x len(input_columns))
    and an expected output vector y, selecting columns by name.
    """

    df = pd.read_csv(path)

    missing = [c for c in [*input_columns, *expected_output] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns not found in {path}: {missing}. Available: {df.columns.tolist()}"
        )

    X = df[input_columns].to_numpy(dtype=np.float64)
    y = df[expected_output].to_numpy(dtype=np.float64)

    return X, y


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

