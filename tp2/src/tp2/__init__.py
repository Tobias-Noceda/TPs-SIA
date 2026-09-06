import argparse
import json

from tqdm import trange

from tp2.models.triangle import Triangle
from tp2.models.polygon_generator import PolygonGenerator

from tp2.algorithms.fitness.fake import ImageFitnessCalculator
from tp2.algorithms.mutation.gene_mutation import gene_mutation

parser = argparse.ArgumentParser(description="Genetic image generator.")

parser.add_argument("-i", "--image", type=str, required=True, help="Image target")
parser.add_argument("-p", "--poligons", dest="n", type=int, required=True, help="Number of triangles")
parser.add_argument("-n", "--population", dest="population", type=int, required=True, help="Population size")
parser.add_argument("-c", "--cycles", dest="cycles", type=int, required=True, help="Number of cycles")
parser.add_argument("-hp", "--hyper", dest="params", type=str, required=True, help="Hyperparameters JSON file")
parser.add_argument("-o", "--output", dest="output", type=str, required=True, help="Output file name")

def main() -> None:
    args = parser.parse_args()

    # with open(args.params, 'r') as file:
    #     data = json.load(file)

    # print(f"Image {args.image} with {args.n} triangles and {data} params")

    fitness_calculator = ImageFitnessCalculator.for_image("", (0, 0), (100, 100))
    mutation_rate = 0.5
    mutation = gene_mutation

    triangles = PolygonGenerator.generate_many_polygons(Triangle, (0, 0), (100, 100), args.n)
    print(f"Initial triangles: {triangles}\n")

    for _ in trange(args.cycles):
        fitness = fitness_calculator.calculate_fitness(triangles)
        triangles = mutation(fitness, mutation_rate)
        print(f"Triangles: {triangles}\n")
