import argparse
import json
import time

from tp2.interfaces.algorithms.breeding import Breeding
from tp2.interfaces.algorithms.fitness import ImageFitnessCalculator
from tp2.interfaces.algorithms.mutation import Mutation
from tp2.interfaces.algorithms.selection import Selection
from tp2.interfaces.algorithms.survival import Survival

from tp2.types.coords import Coords
from tp2.types.individual import Individual

from tqdm import trange

from tp2.rt_vis import RealTimeVis
from tp2.models.triangle import Triangle
from tp2.models.polygon_generator import PolygonGenerator

from tp2.algorithms.breeding.single_point import single_point_crossover
from tp2.algorithms.fitness.delta import DeltaEFitnessCalculator
from tp2.algorithms.fitness.mixed import MixedFitnessCalculator
from tp2.algorithms.fitness.ssim import SSIMFitnessCalculator
from tp2.algorithms.mutation.gene_mutation import gene_mutation
from tp2.algorithms.selection.elite import elite_selection
from tp2.algorithms.survival.additive import additive_survival

parser = argparse.ArgumentParser(description="Genetic image generator.")

parser.add_argument("-i", "--image", type=str, required=True, help="Image target")
parser.add_argument("-p", "--polygons", dest="n", type=int, required=True, help="Number of triangles")
parser.add_argument("-n", "--population", dest="population", type=int, required=True, help="Population size")
parser.add_argument("-c", "--cycles", dest="cycles", type=int, required=True, help="Number of cycles")
parser.add_argument("-conf", "--config", dest="config", type=str, required=True, help="Configuration JSON file")
parser.add_argument("-o", "--output", dest="output", type=str, required=True, help="Output file name")

def main() -> None:
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        data = json.load(file)

    image_fitness: tuple[ImageFitnessCalculator, Coords, tuple[int, ...]] = DeltaEFitnessCalculator.for_image(args.image)
    fitness, size, color = image_fitness

    selection: Selection = elite_selection

    mutation_rate = 0.5
    mutation: Mutation = gene_mutation

    breeding: Breeding = single_point_crossover

    survival: Survival = additive_survival

    initial: list[Individual] = [
        PolygonGenerator.generate_many_polygons(Triangle, (0, 0), size, args.n)
        for _ in range(args.population)
    ]

    population = [
        fitness.calculate_fitness(individual)
        for individual in initial
    ]

    with RealTimeVis(size, color) as vis:
        vis.update(max(population).individual)

        time.sleep(10)

        for generation in trange(args.cycles):
            data["generation"] = generation

            selected = selection(population, data)

            mutated = [
                mutation(individual, mutation_rate)
                for individual in selected
            ]

            bred = [
                breeding(parents)
                for parents in zip(mutated[::2], mutated[1::2])
            ]

            scored = [
                fitness.calculate_fitness(individual)
                for childs in bred
                for individual in childs
            ]

            population = survival(population, scored)

            vis.update(population[0].individual)
