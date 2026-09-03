import argparse
import json

from tp2.models.triangle import Triangle
from tp2.models.polygon_generator import PolygonGenerator

parser = argparse.ArgumentParser(description="Genetic image generator.")

parser.add_argument("-i", "--image", type=str, required=True, help="Image target")
parser.add_argument("-t", "--triangles", dest="n", type=int, required=True, help="Number of triangles")
parser.add_argument("-p", "--hyper", dest="params", type=str, required=True, help="Hyperparameters JSON file")

def main() -> None:
    args = parser.parse_args()

    with open(args.params, 'r') as file:
        data = json.load(file)

    print(f"Image {args.image} with {args.n} triangles and {data} params")

    triangles = PolygonGenerator.generate_many_polygons(Triangle, (0, 0), (100, 100), args.n)

    print(f"Generated triangles:")
    for triangle in triangles:
        print(triangle)
