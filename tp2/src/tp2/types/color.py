from dataclasses import dataclass
import random
from typing import Tuple

MIN_ALPHA = 128

# Represents a color in RGBA format
@dataclass(frozen=True)
class Color:
    r: int  # Red component (0-255)
    g: int  # Green component (0-255)
    b: int  # Blue component (0-255)
    a: int  # Alpha component (0-255)
    
    @staticmethod
    def from_tuple(color_tuple: Tuple[int, int, int, int]) -> Color:
        r, g, b, a = color_tuple
        return Color(r, g, b, a)

    @staticmethod
    def generate_random() -> Color:
        """
        Generates a random color.
        Returns a new Color instance with random values.
        """
        new_r = random.randint(0, 255)
        new_g = random.randint(0, 255)
        new_b = random.randint(0, 255)
        new_a = random.randint(MIN_ALPHA, 255)
        return Color(new_r, new_g, new_b, new_a)

    def pillow(self) -> tuple[int, ...]:
        return (self.r, self.g, self.b, self.a)

    def mutate_randomly(self) -> Color:
        """
        Mutates the color randomly.
        Returns a new Color instance with mutated values.
        """
        return Color.generate_random()

    def mutate_with_fitness(self, fitness: float) -> Color:
        """
        Mutates the color based on a fitness value.
        Fitness value should be a float between 0 and 1, where 1 indicates no mutation and 0 indicates maximum mutation.
        Returns a new Color instance with mutated values based on the fitness.
        """
        if fitness < 0 or fitness > 1:
            raise ValueError("Fitness value must be between 0 and 1.")

        new_r = int(self.r + (random.randint(-255, 255) * (1 - fitness)))
        new_g = int(self.g + (random.randint(-255, 255) * (1 - fitness)))
        new_b = int(self.b + (random.randint(-255, 255) * (1 - fitness)))
        new_a = int(self.a + (random.randint(-255, 255) * (1 - fitness)))

        # Ensure values are within valid ranges
        new_r = max(0, min(255, new_r))
        new_g = max(0, min(255, new_g))
        new_b = max(0, min(255, new_b))
        new_a = max(MIN_ALPHA, min(255, new_a))

        return Color(new_r, new_g, new_b, new_a)

    def __str__(self):
        return f"(r={self.r}, g={self.g}, b={self.b}, a={self.a})"

    def __repr__(self):
        return f"(r={self.r}, g={self.g}, b={self.b}, a={self.a})"