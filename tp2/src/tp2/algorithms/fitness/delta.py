from dataclasses import dataclass
from functools import cache
from typing import override

from tp2.interfaces.fitness_calculator import FitnessCalculator
from tp2.types.individual import Individual

from tp2.models.individual_fitness import IndividualFitness

import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2lab, deltaE_ciede2000

WHITE = (255, 255, 255, 255)

@dataclass(frozen=True)
class DeltaEFitnessCalculator(FitnessCalculator):
    reference_lab: np.ndarray

    width: int
    height: int

    background: tuple[int, ...]

    @override
    @staticmethod
    def for_image(path, lt = (0, 0), rb = None):
        img = Image.open(path).convert("RGB")

        left, top = lt
        right, bottom = img.size if rb is None else rb

        region = (left, top, right, bottom)
        reference = np.asarray(img.crop(region)) / 255.0

        width = right - left
        height = bottom - top

        return DeltaEFitnessCalculator(rgb2lab(reference), width, height, WHITE)

    @cache
    @override
    def calculate_fitness(self, individual) -> IndividualFitness:
        """
        Calculates the fitness of an individual.
        """
        rendered_lab = self._get_individual_lab(
            individual,
            self.width,
            self.height,
            self.background
        )

        de: np.ndarray = deltaE_ciede2000(self.reference_lab, rendered_lab)

        print("Mean ΔE:", de.mean())
        print("Median ΔE:", np.median(de))
        print("95th percentile ΔE:", np.percentile(de, 95))
        print("Max ΔE:", de.max())

        return IndividualFitness(individual, np.median(de))

    @staticmethod
    def _get_individual_lab(individual: Individual, w: int, h: int, bg: tuple[int, ...]) -> np.ndarray:
        img = Image.new("RGBA", (w, h), bg)
        draw = ImageDraw.Draw(img)

        for polygon in individual:
            draw.polygon(polygon.vertices, fill=polygon.color.pillow())

        img.convert("RGB")
        rendered = np.asarray(img) / 255.0

        return rgb2lab(rendered)
