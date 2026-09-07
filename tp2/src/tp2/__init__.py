import argparse
import time
from multiprocess.pool import Pool

from tp2.config import load_config

from tp2.interfaces.algorithms.breeding import Breeding
from tp2.interfaces.algorithms.fitness import ImageFitnessCalculator
from tp2.interfaces.algorithms.mutation import Mutation
from tp2.interfaces.algorithms.selection import Selection
from tp2.interfaces.algorithms.survival import Survival

from tp2.types.coords import Coords
from tp2.types.individual import Individual

from tqdm import trange

from tp2.rt_vis import RealTimeVis
from tp2.output import save_result
from tp2.models.triangle import Triangle
from tp2.models.polygon_generator import PolygonGenerator

parser = argparse.ArgumentParser(description="Genetic image generator.")

parser.add_argument("-i", "--image", type=str, required=True, help="Image target")
parser.add_argument("-p", "--polygons", dest="n", type=int, required=True, help="Number of triangles")
parser.add_argument("-n", "--population", dest="population", type=int, required=False, default=20, help="Population size")
parser.add_argument("-m", "--mutation-rate", dest="mutation_rate", type=float, required=False, default=0.5, help="Mutation rate")
parser.add_argument("-c", "--cycles", dest="cycles", type=int, required=False, default=10000, help="Number of maximum cycles to run")
parser.add_argument("-f", "--fitness", dest="fitness", type=float, required=False, help="Fitness threshold to stop the algorithm")
parser.add_argument("-conf", "--config", dest="config", type=str, required=True, help="Configuration JSON file")
parser.add_argument("-o", "--output", dest="output", type=str, required=True, help="Output file name")
parser.add_argument("-mp", "--multiprocess", dest="mercado_pago", action='store_true', help="Wether to use multiprocessing")

def main() -> None:
    args = parser.parse_args()

    data = load_config(args.config)
    hyperparameters: dict = data["hyperparameters"]

    # print(f"Loaded config: {data}")

    fitness_calculator: type[ImageFitnessCalculator] = data["fitness"].calculator
    image_fitness: tuple[ImageFitnessCalculator, Coords, tuple[int, ...]] = fitness_calculator.for_image(args.image)
    fitness, size, color = image_fitness

    selection: Selection = data["selection"].function

    mutation_rate: float = 0.5 if data["mutation"]["rate"] is None else data["mutation"]["rate"]
    mutation: Mutation = data["mutation"]["algorithm"].function

    breeding: Breeding = data["breeding"].function

    survival: Survival = data["survival"].function

    # print(f"Breeding algorithm: {breeding.__name__}")
    # print(f"Fitness algorithm: {fitness_calculator.__name__}")
    # print(f"Mutation algorithm: {mutation.__name__}")
    # print(f"Selection algorithm: {selection.__name__}")
    # print(f"Survival algorithm: {survival.__name__}")
    # print(f"Hyperparameters: {hyperparameters}")

    initial: list[Individual] = [
        PolygonGenerator.generate_many_polygons(Triangle, (0, 0), size, args.n)
        for _ in range(args.population)
    ]

    population = tuple(sorted([
        fitness.calculate_fitness(individual)
        for individual in initial
    ], reverse=True))

    with RealTimeVis(size, color) as vis, Pool() as pool:
        vis.update(population[0].individual)

        time.sleep(10)
        start = time.perf_counter()

        scores: list[float] = []
        for generation in trange(args.cycles):
            hyperparameters["generation"] = generation

            selected = selection(population, hyperparameters)

            if args.mercado_pago:
                mutated = pool.map(lambda i: mutation(i, mutation_rate), selected)

                bred = pool.map(breeding, zip(mutated[::2], mutated[1::2]))

                flatted = [
                    individual
                    for childs in bred
                    for individual in childs
                ]

                scored = pool.map(fitness.calculate_fitness, flatted)

            else:
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

            scores.append(population[0].score)
            vis.update(population[0].individual)

            if args.fitness is not None and population[0].score >= args.fitness:
                print(f"Fitness threshold reached: {population[0].score} >= {args.fitness}")
                break

        duration = time.perf_counter() - start
        save_result(args.output, population[0], hyperparameters["generation"], scores, duration, size, color)
