from tp2.models.individual_fitness import IndividualFitness

import json
from pathlib import Path

from tp2.types.coords import Coords

def save_result(
    path: Path,
    individual: IndividualFitness,
    generation: int,
    scores: list[float],
    seconds: float,
    size: Coords,
    bg: tuple[int, ...]
):
    with open(Path.cwd() / path, "w") as output:
        polygons = [
            {
                "vertices": triangle.vertices,
                "color": triangle.color.pillow()
            }
            for triangle in individual.individual
        ]

        data = {
            "counter": generation,
            "duration": float(seconds),
            "fitness": float(individual.score),
            "scores": scores,
            "individual": {
                "size": size,
                "background": bg,
                "polygons": polygons,
            }
        }

        output.write(json.dumps(data))
