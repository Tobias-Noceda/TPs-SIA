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
    def __str__(self) -> str:
        """
        Returns a string representation of the polygon.
        This method should be overridden in subclasses to provide specific string representation logic.
        """
        pass
