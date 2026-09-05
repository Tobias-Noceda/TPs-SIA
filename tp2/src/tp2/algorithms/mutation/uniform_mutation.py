import random

from tp2.models.individual_fitness import IndividualFitness
from tp2.types.individual import Individual

def uniform_mutation(individual_fitness: IndividualFitness, mutation_rate: float) -> Individual:
    """
    Applies a mutation to each gene of the given individual based on its mutation rate.
    
    Args:
        individual_fitness (IndividualFitness): The individual along with its fitness score.
        mutation_rate (float): The probability of mutating a gene (0 < mutation_rate <= 1).
        
    Returns:
        Individual: A new individual that has undergone mutation.
    """
    new_genes = list(individual_fitness.individual)
    for i in range(len(new_genes)):
        # if a random number between 0 and 1 is less than the mutation rate, mutate the gene at the current index
        if random.uniform(0, 1) < mutation_rate:
            new_genes[i] = new_genes[i].mutate_with_fitness(individual_fitness.score)

    return new_genes