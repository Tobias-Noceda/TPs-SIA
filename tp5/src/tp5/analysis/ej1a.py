import argparse
import dataclasses
from pathlib import Path
from typing import Any, Dict, List

from tp5.persistence.run_file import load_run
from tp5.visualization.autoencoder import plot_run
from tp5.visualization.sweep import plot_hardest_patterns, plot_study_curves, plot_study_summary

import numpy as np
import pandas as pd
import matplotlib

def _load_runs(runs_dir: Path, goal: int) -> List[Dict[str, Any]]:
    """
    Every run of the sweep, measured against `goal` wrong pixels (the objective of the assignment)
    and not against the goal_pixels each run was trained with: the base config has none (saved as
    -1) and the runs stop by epsilon, so the epoch the goal is reached comes from the history.
    """
    runs = []
    for path in sorted(runs_dir.glob("*__*__s*.npz")):
        study, variant, seed = path.stem.split("__")
        data = np.load(path)
        history = data["history"]
        errors = data["pixel_errors"]
        learned = history[:, 2] <= goal
        runs.append({
            "path": path,
            "study": study,
            "variant": variant,
            "seed": int(seed.removeprefix("s")),
            "epochs": len(history),
            "reached_goal": bool(learned.any()),
            "epochs_to_goal": int(np.argmax(learned)) + 1 if learned.any() else np.nan,
            "final_loss": float(history[-1, 1]),
            "max_pixel_errors": int(errors.max()),
            "mean_pixel_errors": float(errors.mean()),
            "patterns_learned": int(np.sum(errors <= goal)),
            "elapsed": float(data["elapsed"]),
            "history": history,
            "pixel_errors": errors,
            "labels": list(data["labels"]),
            "goal_pixels": goal,
        })
    return runs

def main() -> None:
    parser = argparse.ArgumentParser(description="Ex. 1a.2: summary and plots of the autoencoder sweep.")
    parser.add_argument("-i", "--input", default="res/output/ej1a", help="Sweep output directory (with runs/)")
    parser.add_argument("-g", "--goal", type=int, default=1, help="Wrong pixels allowed per pattern (default 1, the assignment's goal)")
    args = parser.parse_args()
    matplotlib.use("Agg")

    root = Path(args.input)
    runs = _load_runs(root / "runs", args.goal)
    if not runs:
        raise SystemExit(f"No runs found in {root / 'runs'}")

    columns = ["study", "variant", "seed", "epochs", "reached_goal", "epochs_to_goal", "final_loss",
               "max_pixel_errors", "mean_pixel_errors", "patterns_learned", "elapsed"]
    summary = pd.DataFrame([{k: run[k] for k in columns} for run in runs])
    summary.to_csv(root / "summary.csv", index=False)

    aggregate = summary.groupby(["study", "variant"]).agg(
        runs=("seed", "count"),
        success_rate=("reached_goal", "mean"),
        median_epochs_to_goal=("epochs_to_goal", "median"),
        mean_max_pixel_errors=("max_pixel_errors", "mean"),
        mean_patterns_learned=("patterns_learned", "mean"),
        mean_elapsed=("elapsed", "mean"),
    ).reset_index()
    aggregate.to_csv(root / "aggregate.csv", index=False)
    print(aggregate.to_string(index=False))

    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for study in summary["study"].unique():
        study_runs = [run for run in runs if run["study"] == study]
        plot_study_curves(study, study_runs).savefig(plots / f"{study}_curves.png", dpi=130)
        plot_study_summary(study, study_runs).savefig(plots / f"{study}_summary.png", dpi=130)
        matplotlib.pyplot.close("all")

    failed = [run for run in runs if not run["reached_goal"]]
    if failed:
        plot_hardest_patterns(failed).savefig(plots / "hardest_patterns.png", dpi=130)
        matplotlib.pyplot.close("all")

    # Best run: reaches the goal in the fewest epochs; otherwise the fewest wrong pixels
    best = min(runs, key=lambda r: (not r["reached_goal"], r["epochs_to_goal"] if r["reached_goal"] else r["max_pixel_errors"], r["final_loss"]))
    print(f"Best run: {best['path'].name} (goal: {best['reached_goal']}, epochs: {best['epochs']}, max wrong pixels: {best['max_pixel_errors']})")
    # Its figures mark the same goal the sweep is measured against
    plot_run(dataclasses.replace(load_run(str(best["path"])), goal_pixels=args.goal), save_dir=str(plots / "best"))
    print(f"Saved summary.csv, aggregate.csv and plots/ to {root}")
