from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from tp5.interfaces.autoencoder import Autoencoder
from tp5.types.epoch_result import EpochResult
from tp5.types.samples import Samples

@dataclass
class TrainingRun:
    """
    Everything a run produces: the trained model plus what is needed to report, save and plot it.

    Attributes:
        model: The trained autoencoder.
        inputs: Training patterns (samples, pixels).
        labels: Character of each pattern.
        shape: Glyph shape (rows, columns).
        history: One EpochResult per trained epoch.
        goal_pixels: Wrong pixels allowed for a pattern to count as learned.
        seed: Entropy of the SeedSequence (reproduces the run when the config had no seed).
        elapsed: Training wall time in seconds.
        source: The config as written.
    """
    model: Autoencoder
    inputs: Samples
    labels: List[str]
    shape: Tuple[int, int]
    history: Tuple[EpochResult, ...]
    goal_pixels: int
    seed: int
    elapsed: float
    source: Dict[str, Any]
