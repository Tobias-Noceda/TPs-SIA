import random

from tp2.models.individual_fitness import IndividualFitness
from tp2.types.individual import Individual

def gene_mutation(individual_fitness: IndividualFitness, mutation_rate: float) -> Individual:
    """
    Applies a mutation to one gene of the given individual based on its mutation rate.
    
    Args:
        individual_fitness (IndividualFitness): The individual along with its fitness score.
        mutation_rate (float): The probability of mutating a gene (0 < mutation_rate <= 1).
        
    Returns:
        Individual: A new individual that has undergone mutation.
    """
    mutate_index = random.randint(0, len(individual_fitness.individual) - 1)

    new_genes = list(individual_fitness.individual)
    # if a random number between 0 and 1 is less than the mutation rate, mutate the gene at the selected index
    if random.uniform(0, 1) < mutation_rate:
        new_genes[mutate_index] = new_genes[mutate_index].mutate_with_fitness(individual_fitness.score)

    return new_genes