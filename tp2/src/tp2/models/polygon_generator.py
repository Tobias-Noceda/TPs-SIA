from tp2.interfaces.polygon import Polygon
from tp2.types.coords import Coords

class PolygonGenerator():
    @staticmethod
    def generate_polygon(pol: type[Polygon], top_left: Coords, bottom_right: Coords) -> Polygon:
        """
        Generates a polygon with the specified number of sides.
        This method should be overridden in subclasses to provide specific polygon generation logic.
        """
        return pol.generate_random(top_left, bottom_right)

    @staticmethod
    def generate_many_polygons(pol: type[Polygon], top_left: Coords, bottom_right: Coords, count: int) -> tuple[Polygon, ...]:
        """
        Generates multiple polygons with the specified number of sides.
        This method can be used to generate a list of polygons based on the provided coordinates and count.
        """
        return tuple([PolygonGenerator.generate_polygon(pol, top_left, bottom_right) for _ in range(count)])
