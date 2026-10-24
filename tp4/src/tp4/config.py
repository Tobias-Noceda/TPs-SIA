import copy
import json
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from tp4.types.config.config import Config

from tp4.types.config.components import ComponentCriteria
from tp4.types.config.initialization import InitializationFunctions
from tp4.types.config.method import Methods
from tp4.types.config.neighborhood import NeighborhoodFunctions
from tp4.types.config.normalization import NormalizationFunctions
from tp4.types.config.orientation import Orientations
from tp4.types.config.schedule import ScheduleFunctions
from tp4.types.config.similarity import SimilarityFunctions
from tp4.types.config.topology import Topologies

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
            lines.append(_describe(sub, indent + 2, root))

    return "\n".join(lines)

def _load_schema() -> dict[str, Any]:
    with open(SCHEMA_PATH, "r") as file:
        return json.load(file)

def load_config(path: str) -> Config:
    """
    Load the config JSON at `path` and validate it against res/config/config.schema.json.

    Raises ValueError describing the expected structure when validation fails.
    """
    with open(path, "r") as file:
        data = json.load(file)

    return parse_config(data, path)

def parse_config(data: dict[str, Any], path: str = "<memory>") -> Config:
    """
    Validate an already loaded config dict (e.g. built by an experiment) and resolve it.

    `path` only names the config in error messages. Raises ValueError like load_config.
    """
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

    data["method"] = Methods(data["method"])
    data["seed"] = int(data["seed"]) if "seed" in data else None
    # PCA standardizes by itself, so its dataset has no normalization: the default is only a placeholder
    data["dataset"]["normalize"] = NormalizationFunctions(data["dataset"].get("normalize", "z_score")).function()

    if data["method"] is Methods.KOHONEN:
        kohonen = data["kohonen"]
        data["kohonen"] = {
            "grid_size": int(kohonen["grid_size"]),
            "topology": Topologies(kohonen.get("topology", "rectangular")).topology(),
            "similarity": SimilarityFunctions(kohonen.get("similarity", "euclidean")).function(),
            "neighborhood": NeighborhoodFunctions(kohonen.get("neighborhood", {"name": "step"})["name"]).function(),
            "initialization": InitializationFunctions(kohonen.get("initialization", "samples")).function(),
        }
        for schedule in ("learning_rate", "radius"):
            data[schedule] = ScheduleFunctions(data[schedule]["type"]).factory(
                data[schedule]["value"], data[schedule].get("hyperparameters", None)
            )
        data["epochs"] = int(data["epochs"])
    elif data["method"] is Methods.PCA:
        components = data["pca"].get("components", {"criterion": "kaiser"})
        data["pca"] = {
            "components": ComponentCriteria(components["criterion"]).function(components.get("value", None)),
            "orientation": Orientations(data["pca"].get("orientation", "max_loading_positive")).function(),
        }

    return cast(Config, data)
