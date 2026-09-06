from dataclasses import dataclass
from functools import cache
from typing import override

from tp2.interfaces.fitness_calculator import FitnessCalculator


@dataclass(frozen=True)
class ImageFitnessCalculator(FitnessCalculator):
    """
    Calculates the fitness of an image individual based on its similarity to a target image.
    """
    # The target image is represented as a 2D list of RGBA tuples, where each tuple contains the red, green, blue, and alpha components of a pixel.
    target_image: list[list[tuple[int, int, int, float]]]
    image_top_left: tuple[int, int]  # (x, y) coordinates of the top-left corner of the image
    image_bottom_right: tuple[int, int]  # (x, y) coordinates of the bottom-right corner of the image

    @cache
    @override
    def calculate_fitness(self) -> float:
        """
        Calculates the fitness of an image individual based on its similarity to the target image.
        The fitness is calculated as the inverse of the average pixel difference between the individual's image and the target image.
        A higher fitness score indicates a closer match to the target image.
        """
        return 0.5  # Placeholder
