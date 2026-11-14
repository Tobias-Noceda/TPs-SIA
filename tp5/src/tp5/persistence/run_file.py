import json
from pathlib import Path
from typing import Any, Dict, List

from tp5.functions.pixel_error import pixel_errors
from tp5.models.basic_autoencoder import BasicAutoencoder
from tp5.types.training_run import TrainingRun

import numpy as np

def save_run(path: str, run: TrainingRun) -> None:
    """
    Save a run as .npz: config, history, latent codes, reconstructions, wrong pixels per pattern and
    the weights of every layer (encoder_W{i}, encoder_b{i}, decoder_W{i}, decoder_b{i}).
    """
    if not isinstance(run.model, BasicAutoencoder):
        raise TypeError(f"Saving {type(run.model).__name__} is not supported.")

    reconstructions = run.model.reconstruct(run.inputs)
    parameters: Dict[str, Any] = {}
    for name, network in (("encoder", run.model.encoder), ("decoder", run.model.decoder)):
        for i, (weights, bias) in enumerate(network.parameters()):
            parameters[f"{name}_W{i}"] = weights
            parameters[f"{name}_b{i}"] = bias

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        config=json.dumps(run.source),
        history=np.array(run.history, dtype=np.float64).reshape(-1, 4),
        inputs=run.inputs,
        labels=np.array(run.labels),
        shape=np.array(run.shape),
        latent=run.model.encode(run.inputs),
        reconstructions=reconstructions,
        pixel_errors=pixel_errors(reconstructions, run.inputs),
        goal_pixels=run.goal_pixels,
        seed=str(run.seed),
        elapsed=run.elapsed,
        **parameters,
    )

def load_parameters(data: Any, name: str) -> List[Any]:
    """(weights, bias) per layer of the `encoder` or `decoder` saved by save_run."""
    count = sum(1 for key in data.files if key.startswith(f"{name}_W"))
    return [(data[f"{name}_W{i}"], data[f"{name}_b{i}"]) for i in range(count)]
