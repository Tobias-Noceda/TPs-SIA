import copy
import json
from pathlib import Path
from typing import Any, List, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from tp5.types.config.activation import ActivationFunctions
from tp5.types.config.autoencoder import AutoencoderType, LayerDefinition
from tp5.types.config.config import AutoencoderConfig
from tp5.types.config.initialization import InitializationFunctions
from tp5.types.config.learning_rate import LearningRateFunctions
from tp5.types.config.loss import LossFunctions

import numpy as np

# Schema is read at runtime so any edit to this file drives validation automatically.
# @link https://json-schema.org/
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "res" / "config" / "config.schema.json"

def _describe(schema: dict[str, Any], indent: int = 1, root: dict[str, Any] | None = None) -> str:
    """Render a JSON Schema (the subset we use) as an indented structure outline."""
    root = root if root is not None else schema
    if "$ref" in schema:
        # Local references only ("#/$defs/name"); sibling keys (e.g. description) override the target's
        target: Any = root
        for key in schema["$ref"].removeprefix("#/").split("/"):
            target = target[key]
        schema = {**target, **{k: v for k, v in schema.items() if k != "$ref"}}

    pad = "  " * indent
    kind = schema.get("type", "any")
    desc = schema.get("description")
    lines = [f"{pad}{kind}" + (f" — {desc}" if desc else "")]

    bounds = [
        f"minimum {schema['minimum']}" if "minimum" in schema else None,
        f"maximum {schema['maximum']}" if "maximum" in schema else None,
        f"one of {schema['enum']}" if "enum" in schema else None,
    ]
    bounds = [b for b in bounds if b]
    if bounds:
        lines.append(f"{pad}  ({', '.join(bounds)})")

    if kind == "object":
        required = set(schema.get("required", []))
        for name, sub in schema.get("properties", {}).items():
            tag = "required" if name in required else "optional"
            lines.append(f"{pad}  - {name} ({tag}):")
            lines.append(_describe(sub, indent + 2, root))

    return "\n".join(lines)

def _load_schema() -> dict[str, Any]:
    with open(SCHEMA_PATH, "r") as file:
        return json.load(file)

def _activation(definition: dict[str, Any]):
    return ActivationFunctions(definition["name"]).functions(definition.get("hyperparameters", None))

def _layer(layer: list[dict[str, Any]] | dict[str, Any]) -> LayerDefinition:
    # Compact form: { neurons: n, activation_function } = n neurons with the same activation
    if isinstance(layer, dict):
        return [_activation(layer["activation_function"])] * int(layer["neurons"])
    return [_activation(neuron["activation_function"]) for neuron in layer]

def _network(network: dict[str, Any]) -> List[LayerDefinition]:
    return [_layer(layer) for layer in network["arquitecture"]]

def load_config(path: str) -> AutoencoderConfig:
    """
    Load the config JSON at `path` and validate it against res/config/config.schema.json.

    Raises ValueError describing the expected structure when validation fails.
    """
    with open(path, "r") as file:
        data = json.load(file)

    return parse_config(data, path)

def parse_config(data: dict[str, Any], path: str = "<memory>") -> AutoencoderConfig:
    """
    Validate an already loaded config dict (e.g. built by an experiment) and resolve it.

    `path` only names the config in error messages. Raises ValueError like load_config.
    """
    source = copy.deepcopy(data)
    data = copy.deepcopy(data)

    schema = _load_schema()
    errors = sorted(Draft202012Validator(schema).iter_errors(data), key=str)

    if errors:
        primary = best_match(errors)

        def loc(err: Any) -> str:
            return "/".join(map(str, err.absolute_path)) or "<root>"

        violations = "\n".join(f"  - at '{loc(e)}': {e.message}" for e in errors)
        raise ValueError(
            f"Invalid config file '{path}':\n"
            f"{violations}\n\n"
            f"First problem — at '{loc(primary)}': {primary.message}\n\n"
            f"Expected structure:\n{_describe(schema)}"
        )

    autoencoder = data["autoencoder"]
    data["autoencoder"] = {
        "type": AutoencoderType(autoencoder["type"]),
        "encoder": _network(autoencoder["encoder"]),
        "decoder": _network(autoencoder["decoder"]),
    }
    data["loss_function"] = LossFunctions(data.get("loss_function", "mse")).functions()
    data["learning_rate"] = LearningRateFunctions(data["learning_rate"]["type"]).factory(
        data["learning_rate"]["value"], data["learning_rate"].get("hyperparameters", None)
    )
    data["epochs"] = int(data["epochs"])
    data["batch_size"] = int(data.get("batch_size", 1))
    data["epsilon"] = np.float64(data.get("epsilon", 1e-6))
    # No goal: training stops only by epsilon (or the epochs limit)
    data["goal_pixels"] = int(data["goal_pixels"]) if data.get("goal_pixels") is not None else None
    data["stop_at_goal"] = bool(data.get("stop_at_goal", True))
    data["seed"] = int(data["seed"]) if "seed" in data else None
    data["initialization"] = InitializationFunctions(data.get("initialization", "random")).function()
    data["source"] = source

    return cast(AutoencoderConfig, data)
