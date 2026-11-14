import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tp5.config import parse_config
from tp5.persistence.run_file import load_parameters
from tp5.runners.basic import build_basic_autoencoder
from tp5.types.training_run import TrainingRun
from tp5.visualization.autoencoder import plot_run
from tp5.visualization.sweep import plot_hardest_patterns, plot_study_curves, plot_study_summary

import numpy as np
import pandas as pd
import matplotlib
from numpy.random import RandomState

def _load_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(runs_dir.glob("*__*__s*.npz")):
        study, variant, seed = path.stem.split("__")
        data = np.load(path)
        history = data["history"]
        errors = data["pixel_errors"]
        goal = int(data["goal_pixels"])
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

def _rebuild(path: Path) -> TrainingRun:
    """TrainingRun of a saved run: the model is rebuilt from its config and the saved weights."""
    data = np.load(path)
    source = json.loads(str(data["config"]))
    config = parse_config(source, str(path))
    inputs = data["inputs"]
    model = build_basic_autoencoder(config, inputs.shape[1], RandomState(0))
    model.encoder.load_parameters(load_parameters(data, "encoder"))
    model.decoder.load_parameters(load_parameters(data, "decoder"))
    history = tuple((int(e), np.float64(l), int(m), int(p)) for e, l, m, p in data["history"])
    shape = data["shape"]
    return TrainingRun(
        model, inputs, [str(label) for label in data["labels"]], (int(shape[0]), int(shape[1])),
        history, int(data["goal_pixels"]), int(str(data["seed"])), float(data["elapsed"]), source,
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Ex. 1a.2: summary and plots of the autoencoder sweep.")
    parser.add_argument("-i", "--input", default="res/output/ej1a", help="Sweep output directory (with runs/)")
    args = parser.parse_args()
    matplotlib.use("Agg")

    root = Path(args.input)
    runs = _load_runs(root / "runs")
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
    plot_run(_rebuild(best["path"]), save_dir=str(plots / "best"))
    print(f"Saved summary.csv, aggregate.csv and plots/ to {root}")
