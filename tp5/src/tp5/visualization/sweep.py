from typing import Any, Dict, List

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def _padded(histories: List[npt.NDArray[np.float64]], column: int) -> npt.NDArray[np.float64]:
    """(runs, epochs) matrix of one history column; runs that stopped early keep their last value."""
    length = max(len(h) for h in histories)
    return np.array([np.pad(h[:, column], (0, length - len(h)), mode="edge") for h in histories])

def plot_study_curves(study: str, runs: List[Dict[str, Any]]) -> Figure:
    """Median (line) and min–max over seeds (band) of the loss and the max wrong pixels per variant."""
    fig, (ax_loss, ax_pixels) = plt.subplots(1, 2, figsize=(12, 4.5))
    for variant in dict.fromkeys(run["variant"] for run in runs):
        histories = [run["history"] for run in runs if run["variant"] == variant]
        for ax, column in ((ax_loss, 1), (ax_pixels, 2)):
            values = _padded(histories, column)
            epochs = np.arange(1, values.shape[1] + 1)
            line, = ax.plot(epochs, np.median(values, axis=0), label=variant)
            ax.fill_between(epochs, values.min(axis=0), values.max(axis=0), color=line.get_color(), alpha=0.15)

    goal = runs[0]["goal_pixels"]
    ax_pixels.axhline(goal, color="gray", linestyle="--", linewidth=1)
    for ax, title in ((ax_loss, "Loss"), (ax_pixels, "Max wrong pixels")):
        ax.set_xscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    ax_loss.set_yscale("log")
    ax_pixels.legend(title=study)
    fig.suptitle(f"{study}: median and min–max over seeds")
    fig.tight_layout()
    return fig

def plot_study_summary(study: str, runs: List[Dict[str, Any]]) -> Figure:
    """Success rate, epochs to reach the goal and learned patterns per variant."""
    variants = list(dict.fromkeys(run["variant"] for run in runs))
    by_variant = {v: [run for run in runs if run["variant"] == v] for v in variants}
    total = len(runs[0]["pixel_errors"])

    fig, (ax_rate, ax_epochs, ax_learned) = plt.subplots(1, 3, figsize=(14, 4))
    ax_rate.bar(variants, [np.mean([r["reached_goal"] for r in by_variant[v]]) for v in variants])
    ax_rate.set_ylim(0, 1)
    ax_rate.set_title("Runs reaching the goal")

    epochs = [[r["epochs_to_goal"] for r in by_variant[v] if r["reached_goal"]] for v in variants]
    ax_epochs.boxplot([e if e else [np.nan] for e in epochs], tick_labels=variants)
    ax_epochs.set_yscale("log")
    ax_epochs.set_title("Epochs to the goal (successful runs)")

    ax_learned.boxplot([[r["patterns_learned"] for r in by_variant[v]] for v in variants], tick_labels=variants)
    ax_learned.axhline(total, color="gray", linestyle="--", linewidth=1)
    ax_learned.set_title(f"Patterns learned (of {total})")

    for ax in (ax_rate, ax_epochs, ax_learned):
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(study)
    fig.tight_layout()
    return fig

def plot_hardest_patterns(failed: List[Dict[str, Any]]) -> Figure:
    """Among the runs that did not learn everything, how often each pattern was not learned."""
    labels = failed[0]["labels"]
    goal = failed[0]["goal_pixels"]
    frequency = np.mean([run["pixel_errors"] > goal for run in failed], axis=0)
    order = np.argsort(-frequency)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar([labels[i] for i in order], frequency[order])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of failed runs")
    ax.set_title(f"Patterns not learned (> {goal} wrong pixels) in the {len(failed)} runs that missed the goal")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
