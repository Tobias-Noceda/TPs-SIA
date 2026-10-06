import json
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from tp3.types.config.classification import ClassificationFunctions
from tp3.types.config.config import PerceptronConfig

from tp3.types.config.activation import ActivationFunctions
from tp3.types.config.loss import LossFunctions
from tp3.types.config.learning_rate import LearningRateFunctions
from tp3.types.config.perceptron import PerceptronType

import numpy as np

# Schema is read at runtime so any edit to this file drives validation automatically.
# @link https://json-schema.org/
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "res" / "config" / "config.schema.json"

def _describe(schema: dict[str, Any], indent: int = 1) -> str:
    """Render a JSON Schema (the subset we use) as an indented structure outline."""
    pad = "  " * indent
    kind = schema.get("type", "any")
    desc = schema.get("description")
    lines = [f"{pad}{kind}" + (f" — {desc}" if desc else "")]

    bounds = [
        f"minimum {schema['minimum']}" if "minimum" in schema else None,
        f"maximum {schema['maximum']}" if "maximum" in schema else None,
        f"multiple of {schema['multipleOf']}" if "multipleOf" in schema else None,
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
            lines.append(_describe(sub, indent + 2))

    return "\n".join(lines)

def _load_schema() -> dict[str, Any]:
    with open(SCHEMA_PATH, "r") as file:
        return json.load(file)

def load_config(path: str) -> PerceptronConfig:
    """
    Load the config JSON at `path` and validate it against res/config/config.schema.json.

    Raises ValueError describing the expected structure when validation fails.
    """
    with open(path, "r") as file:
        data = json.load(file)

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

    data["loss_function"] = LossFunctions(data["loss_function"]).functions()
    data["learning_rate"] = LearningRateFunctions(data["learning_rate"]["type"]).factory(
        data["learning_rate"]["value"], data["learning_rate"].get("hyperparameters", None)
    )
    data["epochs"] = int(data["epochs"])
    data["batch_size"] = int(data.get("batch_size", 1))
    data["epsilon"] = np.float64(data.get("epsilon", 1e-6))
    data["seed"] = int(data["seed"]) if "seed" in data else None

    activation_function_hyper = data["perceptron"].get("activation_function", {}).get("hyperparameters", None)
    data["perceptron"] = {
        "type": PerceptronType(data["perceptron"]["type"]),
        "activation_function": ActivationFunctions(data["perceptron"]["activation_function"]["name"]).functions(activation_function_hyper) if "activation_function" in data["perceptron"] else None,
        "arquitecture": [
            [ActivationFunctions(func["activation_function"]["name"]).functions(func["activation_function"].get("hyperparameters", None)) for func in layer]
            for layer in data["perceptron"].get("arquitecture", [])
        ] if "arquitecture" in data["perceptron"] else None
    }
    data["classification_function"] = ClassificationFunctions(data["classification_function"]).function()
    data["test_in_train"] = bool(data.get("test_in_train", False))

    return cast(PerceptronConfig, data)
