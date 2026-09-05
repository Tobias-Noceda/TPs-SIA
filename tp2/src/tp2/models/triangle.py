from dataclasses import dataclass
import random
from typing import override

from tp2.types.coords import Coords
from tp2.types.color import Color

from tp2.interfaces.polygon import Polygon

@dataclass(frozen=True)
class Triangle(Polygon):
    top_left: Coords
    bottom_right: Coords

    def __init__(self, vertices: list[Coords], color: Color, top_left: Coords, bottom_right: Coords):
        super().__init__(vertices, color)
        object.__setattr__(self, 'top_left', top_left)
        object.__setattr__(self, 'bottom_right', bottom_right)

    
    @staticmethod
    @override
    def generate_random(top_left: Coords, bottom_right: Coords) -> Polygon:
        vertices = []
        for _ in range(3):
            x = random.randint(top_left[0], bottom_right[0])
            y = random.randint(top_left[1], bottom_right[1])
            vertices.append((x, y))

        color = Color.generate_random()
        
        return Triangle(vertices, color, top_left, bottom_right)

    @staticmethod
    @override
    def generate_random_with_color(top_left: Coords, bottom_right: Coords, color: Color) -> Polygon:
        vertices = []
        for _ in range(3):
            x = random.randint(top_left[0], bottom_right[0])
            y = random.randint(top_left[1], bottom_right[1])
            vertices.append((x, y))

        return Triangle(vertices, color, top_left, bottom_right)

    @override
    def mutate_randomly(self, coordinates: bool = True, color: bool = True) -> Polygon:
        new_vertices = list(self.vertices)
        new_color = self.color
        if coordinates:
            new_vertices = []
            for vertex in self.vertices:
                new_x = random.randint(self.top_left[0], self.bottom_right[0])
                new_y = random.randint(self.top_left[1], self.bottom_right[1])
                new_vertices.append((new_x, new_y))

        if color:
            new_color = self.color.mutate_randomly()
            
        return Triangle(new_vertices, new_color, self.top_left, self.bottom_right)
    
    def mutate_with_fitness(self, fitness: float, coordinates: bool = True, color: bool = True) -> Polygon:
        if fitness < 0 or fitness > 1:
            raise ValueError("Fitness value must be between 0 and 1.")

        if fitness == 1:
            return self  # No mutation if fitness is 1
        if fitness == 0:
            return Triangle.generate_random(self.top_left, self.bottom_right)  # Maximum mutation if fitness is 0

        new_vertices = list(self.vertices)
        new_color = self.color

        if coordinates:
            new_vertices = []
            canvas_width = self.bottom_right[0] - self.top_left[0]
            canvas_height = self.bottom_right[1] - self.top_left[1]
            for vertex in self.vertices:
                x_range = (vertex[0] - canvas_height * (1 - fitness), vertex[0] + canvas_height * (1 - fitness))
                y_range = (vertex[1] - canvas_width * (1 - fitness), vertex[1] + canvas_width * (1 - fitness))
                new_x = random.randint(int(max(self.top_left[0], x_range[0])), int(min(self.bottom_right[0], x_range[1])))
                new_y = random.randint(int(max(self.top_left[1], y_range[0])), int(min(self.bottom_right[1], y_range[1])))
                new_vertices.append((new_x, new_y))

        if color:
            new_color = self.color.mutate_with_fitness(fitness)

        return Triangle(new_vertices, new_color, self.top_left, self.bottom_right)

    @override
    def __str__(self) -> str:
        vertices_str = ', '.join(f'({v[0]}, {v[1]})' for v in self.vertices)
        color_str = f'{self.color}'
        return f'Triangle: Vertices: [{vertices_str}], Color: {color_str}'

    @override
    def __repr__(self) -> str:
        return self.__str__()
