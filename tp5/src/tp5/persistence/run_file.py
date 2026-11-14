import json
from pathlib import Path
from typing import Any, Dict, List

from tp5.config import parse_config
from tp5.functions.pixel_error import pixel_errors
from tp5.models.basic_autoencoder import BasicAutoencoder
from tp5.runners.basic import build_basic_autoencoder
from tp5.types.config.autoencoder import AutoencoderType
from tp5.types.image_shape import ImageShape
from tp5.types.training_run import TrainingRun

import numpy as np
from numpy.random import RandomState

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

def load_run(path: str) -> TrainingRun:
    """
    Rebuild a run saved by save_run without training: the model comes from the config stored in the
    file and its weights are replaced by the saved ones.
    """
    data = np.load(path)
    source = json.loads(str(data["config"]))
    config = parse_config(source, path)

    autoencoder_type = config["autoencoder"]["type"]
    if autoencoder_type is not AutoencoderType.BASIC:
        raise NotImplementedError(f"Loading a '{autoencoder_type.value}' autoencoder is not implemented yet.")

    inputs = data["inputs"]
    # The random generator only draws the initial weights, which are overwritten right away
    model = build_basic_autoencoder(config, inputs.shape[1], RandomState(0))
    model.encoder.load_parameters(load_parameters(data, "encoder"))
    model.decoder.load_parameters(load_parameters(data, "decoder"))

    shape = [int(value) for value in data["shape"]]
    # Files saved before images had channels store (height, width) only
    image_shape = ImageShape(shape[0], shape[1], shape[2] if len(shape) > 2 else 1)
    history = tuple((int(e), np.float64(loss), int(m), int(p)) for e, loss, m, p in data["history"])

    return TrainingRun(
        model,
        inputs,
        [str(label) for label in data["labels"]],
        image_shape,
        history,
        int(data["goal_pixels"]),
        int(str(data["seed"])),
        float(data["elapsed"]),
        source,
    )
