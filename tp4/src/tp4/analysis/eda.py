import argparse
from ctypes import cast
from pathlib import Path

from tp4.config import load_config
from tp4.functions.linear_algebra.eigen import sorted_eigen
from tp4.functions.normalization.z_score import fit_z_score, z_score
from tp4.functions.statistics.correlation import correlation_matrix
from tp4.types.config.config import PCAConfig
from tp4.visualization.pca import plot_heatmap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Exploratory data analysis of the dataset of a config: statistics, data checks and plots.
# The checks look at every numeric column of the CSV (not only the configured inputs), so they can
# justify which columns are left out.
parser = argparse.ArgumentParser(description="Exploratory data analysis of a config's dataset.")

parser.add_argument("-c", "--config", dest="config", type=str, required=True, help="Configuration JSON file (its dataset is analyzed)")
parser.add_argument("-o", "--output", dest="output", type=str, default="res/output/eda", help="Output directory")
parser.add_argument("--threshold", dest="threshold", type=float, default=0.9, help="|r| from which a pair of columns is reported as highly correlated")

# Columns documented as computed from others: (column, formula over the dataframe)
DERIVED = {
    "price_ars": lambda df: 1500 * df["price_usd"],
    "total_engagement_per_1000": lambda df: df["likes_per_1000"] + df["comments_per_1000"] + df["saves_per_1000"] + df["shares_per_1000"],
    "estimated_argentina_reach": lambda df: df["average_reach"] * df["argentina_audience_pct"] / 100,
    "organic_posts_pct": lambda df: 100 - df["sponsored_posts_pct"],
}

# Groups of percentages that may add up to 100 (then any one of them is redundant given the rest)
COMPOSITIONS = {
    "topic": ["topic_football_pct", "topic_makeup_pct", "topic_dogs_pct", "topic_art_pct"],
    "content": ["content_tutorial_pct", "content_review_pct", "content_entertainment_pct", "content_showcase_pct"],
    "sponsorship": ["sponsored_posts_pct", "organic_posts_pct"],
}

# Eigenvalues of the correlation matrix below this are an exact linear dependency between inputs
_RANK_TOLERANCE = 1e-8

def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per numeric column: count, missing, mean, std, quartiles, coefficient of variation and skewness."""
    numeric = df.select_dtypes("number")
    table = numeric.describe().T
    table.insert(1, "missing", numeric.isna().sum())
    table["cv"] = table["std"] / table["mean"].abs().replace(0, np.nan)
    table["skew"] = numeric.skew()
    return table

def checks(df: pd.DataFrame, inputs: list[str], threshold: float) -> None:
    numeric = df.select_dtypes("number")

    print(f"Rows: {len(df)}, numeric columns: {numeric.shape[1]}, inputs: {len(inputs)}")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    for column in df.select_dtypes(exclude="number").columns:
        print(f"Duplicated '{column}': {int(df[column].duplicated().sum())}")

    constant = [c for c in numeric.columns if numeric[c].nunique() <= 1]
    print(f"Constant columns: {constant or 'none'}")

    print("Derived columns (max |column − formula|):")
    for column, formula in DERIVED.items():
        if column in df.columns:
            deviation = float((df[column] - formula(df)).abs().max())
            print(f"  {column:<28} {deviation:.3g}{'  (in inputs!)' if column in inputs else ''}")

    print("Compositions (sum per row, min … max):")
    for name, columns in COMPOSITIONS.items():
        present = [c for c in columns if c in df.columns]
        if present:
            total = df[present].sum(axis=1)
            used = sum(c in inputs for c in present)
            print(f"  {name:<12} {total.min():.4g} … {total.max():.4g}  ({used}/{len(present)} columns in inputs)")

    correlation = numeric.corr()
    print(f"Pairs with |r| ≥ {threshold}:")
    columns = correlation.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            r = correlation.iloc[i, j]
            if abs(r) >= threshold: # type: ignore
                both = columns[i] in inputs and columns[j] in inputs
                print(f"  {r:+.4f}  {columns[i]} ~ {columns[j]}{'  (both in inputs)' if both else ''}")

    # Same correlation matrix the PCA uses: a zero eigenvalue means one input is a linear combination of others
    mean, std = fit_z_score(df[inputs].to_numpy(dtype=np.float64))
    eigenvalues, eigenvectors = sorted_eigen(correlation_matrix(z_score(df[inputs].to_numpy(dtype=np.float64), mean, std)))
    dependent = np.flatnonzero(np.abs(eigenvalues) < _RANK_TOLERANCE)
    print(f"Exact linear dependencies among the inputs: {len(dependent)}")
    for k in dependent:
        involved = [f"{eigenvectors[i, k]:+.2f} {inputs[i]}" for i in np.argsort(-np.abs(eigenvectors[:, k])) if abs(eigenvectors[i, k]) > 0.1]
        print(f"  {', '.join(involved)}")

def plots(df: pd.DataFrame, inputs: list[str], directory: Path) -> None:
    raw = df[inputs].to_numpy(dtype=np.float64)
    mean, std = fit_z_score(raw)
    standardized = z_score(raw, mean, std)

    for name, values, title in [
        ("boxplot_raw", raw, "Inputs, raw values"),
        ("boxplot_standardized", standardized, "Inputs, standardized (z-score)"),
    ]:
        figure, ax = plt.subplots(figsize=(max(10, 0.45 * len(inputs)), 5))
        ax.boxplot(values, tick_labels=inputs, patch_artist=True,
                   boxprops=dict(facecolor="#b7d3f6", edgecolor="#256abf"), medianprops=dict(color="#c0392f"),
                   whiskerprops=dict(color="#256abf"), capprops=dict(color="#256abf"),
                   flierprops=dict(marker="o", markersize=3, markerfacecolor="#256abf", markeredgecolor="none"))
        ax.tick_params(axis="x", rotation=60, labelsize=8)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.savefig(directory / f"{name}.png", dpi=150)
        plt.close(figure)

    figure = plot_heatmap(correlation_matrix(standardized), inputs, inputs, "Correlation matrix of the inputs")
    figure.savefig(directory / "correlation.png", dpi=150)
    plt.close(figure)

    columns = 5
    rows = int(np.ceil(len(inputs) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.4 * rows), squeeze=False)
    for ax, column, values in zip(axes.flat, inputs, raw.T):
        ax.hist(values, bins=15, color="#2a78d6", edgecolor="white")
        ax.set_title(column, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(inputs):]:
        ax.set_axis_off()
    figure.tight_layout()
    figure.savefig(directory / "histograms.png", dpi=150)
    plt.close(figure)

def main():
    args = parser.parse_args()
    _, config = load_config(args.config)
    if "dataset" not in config:
        raise ValueError(f"Config {args.config} has no dataset to analyze")
    
    dataset = config["dataset"]
    inputs = dataset["columns"]["input"]

    df = pd.read_csv(dataset["path"])
    missing = [c for c in inputs if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in {dataset['path']}: {missing}")

    directory = Path(args.output)
    (directory / "plots").mkdir(parents=True, exist_ok=True)

    table = summary(df)
    table.to_csv(directory / "summary.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).to_string())
    print()

    checks(df, inputs, args.threshold)
    plots(df, inputs, directory / "plots")
    print(f"\nSaved {directory / 'summary.csv'} and {directory / 'plots'}")
