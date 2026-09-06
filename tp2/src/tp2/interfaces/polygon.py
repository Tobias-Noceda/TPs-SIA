from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Sequence

from tp2.types.coords import Coords
from tp2.types.color import Color

@dataclass(frozen=True)
class Polygon(ABC):
    vertices: Sequence[Coords]
    color: Color

    @staticmethod
    @abstractmethod
    def generate_random(top_left: Coords, bottom_right: Coords) -> Polygon:
        """
        Generates a random polygon.
        This method should be overridden in subclasses to provide specific random polygon generation logic.
        """
        pass

    @staticmethod
    @abstractmethod
    def generate_random_with_color(top_left: Coords, bottom_right: Coords, color: Color) -> Polygon:
        """
        Generates a random polygon with the specified color.
        This method can be used to generate a polygon with a specific color based on the provided coordinates.
        """
        pass

    def get_vertices(self) -> Sequence[Coords]:
        """
        Returns the vertices of the polygon as a sequence of Coords.
        This method should be overridden in subclasses to provide specific vertex information.
        """
        return self.vertices

    def get_color(self) -> Color:
        """
        Returns the color of the polygon.
        This method can be overridden in subclasses to provide specific color information.
        """
        return self.color

    @abstractmethod
    def mutate_randomly(self, coordinates: bool = True, color: bool = True) -> Polygon:
        """
        Mutates the polygon randomly.
        This method should be overridden in subclasses to provide specific mutation logic.
        """
        pass

    @abstractmethod
    def mutate_with_fitness(self, fitness: float, coordinates: bool = True, color: bool = True) -> Polygon:
        """
        Mutates the polygon taking current status as reference and based on a fitness value.
        Fitness value should be a float between 0 and 1, where 1 indicates no mutation and 0 indicates maximum mutation.
        This method should be overridden in subclasses to provide specific mutation logic based on fitness.
        """
        pass

    # @abstractmethod
    # def mutate_color_randomly(self) -> Polygon:
    #     """
    #     Mutates the color of the polygon randomly.
    #     This method should be overridden in subclasses to provide specific color mutation logic.
    #     """
    #     pass

    # @abstractmethod
    # def mutate_color_with_fitness(self, fitness: float) -> Polygon:
    #     """
    #     Mutates the color of the polygon taking current status as reference and based on a fitness value.
    #     Fitness value should be a float between 0 and 1, where 1 indicates no mutation and 0 indicates maximum mutation.
    #     This method can be overridden in subclasses to provide specific color mutation logic based on fitness.
    #     """
    #     return self.mutate_color_randomly()

    # @abstractmethod
    # def mutate_vertices_randomly(self) -> Polygon:
    #     """
    #     Mutates the vertices of the polygon randomly.
    #     This method should be overridden in subclasses to provide specific vertex mutation logic.
    #     """
    #     pass

    # @abstractmethod
    # def mutate_vertices_with_fitness(self, fitness: float) -> Polygon:
    #     """
    #     Mutates the vertices of the polygon taking current status as reference and based on a fitness value.
    #     Fitness value should be a float between 0 and 1, where 1 indicates no mutation and 0 indicates maximum mutation.
    #     This method should be overridden in subclasses to provide specific vertex mutation logic based on fitness.
    #     """
    #     pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Returns a string representation of the polygon.
        This method should be overridden in subclasses to provide specific string representation logic.
        """
        pass
