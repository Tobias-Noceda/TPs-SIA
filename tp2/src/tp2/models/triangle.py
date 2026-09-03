from dataclasses import dataclass
import random
from typing import override

from tp2.types.coords import Coords
from tp2.types.color import Color

from tp2.interfaces.polygon import Polygon

@dataclass(frozen=True)
class Triangle(Polygon):
    
    @staticmethod
    @override
    def generate_random(top_left: Coords, bottom_right: Coords) -> Polygon:
        vertices = []
        for _ in range(3):
            x = random.randint(top_left[0], bottom_right[0])
            y = random.randint(top_left[1], bottom_right[1])
            vertices.append((x, y))

        color = [random.randint(0, 255) for _ in range(3)]
        hue = random.uniform(0, 1)
        
        return Triangle(vertices, (color[0], color[1], color[2], hue))

    @staticmethod
    @override
    def generate_random_with_color(top_left: Coords, bottom_right: Coords, color: Color) -> Polygon:
        vertices = []
        for _ in range(3):
            x = random.randint(top_left[0], bottom_right[0])
            y = random.randint(top_left[1], bottom_right[1])
            vertices.append((x, y))

        return Triangle(vertices, color)

    @override
    def __str__(self) -> str:
        vertices_str = ', '.join(f'({v[0]}, {v[1]})' for v in self.vertices)
        color_str = f'({self.color[0]}, {self.color[1]}, {self.color[2]}, {self.color[3]:.2f})'
        return f'Triangle: Vertices: [{vertices_str}], Color: {color_str}'
