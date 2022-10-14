import numpy as np
import pyDOE


def init_pop(
    pop_size,
    n_level,
    chromosome_type,
    initialization="lhs",
    initial_guess=np.array([]),
):
    """
    The initialisation of the population.

    This function constructs the first generation used during the IGA.

    Parameters
    ----------
    pop_size : int
        The size of the population to be created.


    n_level : list
        The number of possible choices/values that each gene can be.

    chromosome_type : class
        The type of the class keeping the chromosome inside. The class can be
        user made but must inherit from the stimuli.Stimuli class.

    initialization : string
        Population initalization method: 'lhs' (Latin Hypercube Sample) or
        'random'.
        Later improvement should include a space filling-like technique in the
        meta model space, to maximize perceptual diversity.

    Initial_guess : numpy array
        Initial guess population. Can be smaller than default population size.



    Returns
    -------
    numpy array
        A numpy array containing the population.
    """
    gene_size = len(n_level)

    n_guess = initial_guess.shape[0]

    pop = np.array([chromosome_type(n_level) for i in range(pop_size)])
    if initialization == "lhs" and n_guess < pop_size:
        # The popsize for the lhs is the initial pop size minus the number of
        # guesses
        lhs = pyDOE.lhs(gene_size, pop_size - n_guess, criterion="maximin")
        lhs_discretized = np.empty_like(lhs, dtype=int)
        for j in range(gene_size):
            n_level_gene = n_level[j]
            for i in range(pop_size - n_guess):
                lhs_discretized[i, j] = np.sum(
                    [
                        (
                            k / n_level_gene < lhs[i, j]
                            and lhs[i, j] < (k + 1) / n_level_gene
                        )
                        * k
                        for k in range(n_level_gene)
                    ]
                )
        for idx in range(lhs_discretized.shape[0]):
            indiv = pop[idx]
            indiv.set_chromosome(lhs_discretized[idx, :])
    # pop = np.array([chromosome_type(n_level) for i in range(pop_size)])
    # return np.array([chromosome_type(n_level) for i in range(pop_size)])
    if n_guess > 0:
        pop = insert_initial_guess(pop, initial_guess)
    return pop


def insert_initial_guess(pop, guess_pop):
    out_pop = np.copy(pop)
    if guess_pop.shape[0] > pop.shape[0]:
        raise ValueError("The guess pop is larger than pop size.")
    # Fill from the last index backwards
    for idx in range(guess_pop.shape[0]):
        indiv = out_pop[-idx - 1]
        indiv.set_chromosome(guess_pop[idx, :])

    return out_pop


def calc_fitness(pop, fitness_noise_std=0):
    """
    Calculate the fitness for each chromosome in the population.

    Parameters
    ----------
    pop : numpy array
        The population.

    fitness_noise_std : float
        Standard deviation of the noise to add to the fitness value. This is
        to simulate a human evaluation. For more realism, this should be
        dependent on the sound.

    Returns
    -------
    numpy array
        A numpy array containing the fitness scores for each chromosome.
        The fitness = 1 if the sound has the same score as the best attainable
        score in the model, = 0 if is the opposite score.
    """

    fitness = np.array([pop[i].get_score() for i in range(pop.shape[0])])
    # min_score = 132
    # max_score = 660
    # # min_score = 0
    # # max_score = 24
    # fitness = (max_score - fitness) / (max_score - min_score)
    if fitness_noise_std > 0:
        for idx in range(fitness.shape[0]):
            noise = np.random.normal(0, fitness_noise_std)
            if fitness[idx] + noise < 0:
                fitness[idx] = 0
            elif fitness[idx] + noise > 1:
                fitness[idx] = 1
            else:
                fitness[idx] = fitness[idx] + noise
    return fitness


def binary_tournament(population, fitness):
    """
    This function performs a binary tournament to select a parent from the
    input population.

    Parameters
    ----------
    population : numpy array
        The population.

    fitness : numpy array
        The fitness for each chromosome of the population.

    Returns
    -------
    winner : Stimuli
        The winner individual.


    """
    # Create list of competitors in tournament
    competitor_list = np.arange(fitness.shape[0])
    # Randomly select first competitor
    competitor1 = np.random.choice(competitor_list)
    # Remove first competitor from list
    competitor_list = competitor_list[competitor_list != competitor1]
    # Select second competitor
    competitor2 = np.random.choice(competitor_list)
    # The winner is the competitor with the lowest fitness
    if fitness[competitor1] < fitness[competitor2]:
        winner = population[competitor1].copy()
        winner_idx = competitor1
    else:
        winner = population[competitor2].copy()
        winner_idx = competitor2
    return winner, winner_idx


def uniform_crossover(parent1, parent2, bias=-1):
    """
    Uniform crossover operation.
    Makes a uniform crossover between the two input parents. The number of
    genes that are taken from each parent depends on a bias towards the first
    parent.

    Parameters
    ----------
    parent1 : Stimuli
        First parent.

    parent1 : Stimuli
        Second parent.

    bias : float
        Percent of genes going from 1st parent to 1st child:
            - child1 = bias(%) parent1 + (1 - bias)(%) parent2
            - child2 = (1 - bias)(%) parent1 + bias(%) parent2

    Returns
    -------
    child : Stimuli
        Generated child

    """

    # If bias is not specified, choose randomly.
    if bias < 0:
        bias = np.random.uniform() * 0.5
    # Get parents' chromosome
    parent1_chromosome = parent1.get_chromosome()
    parent2_chromosome = parent2.get_chromosome()
    # Create child chromosome by copying each parent
    child1_chromosome = parent1_chromosome.copy()
    child2_chromosome = parent2_chromosome.copy()
    # Get number of genes in a chromosome
    n_genes = parent1_chromosome.shape[0]
    # Create gene list
    gene_list = np.arange(n_genes)
    # Select how many genes will be taken from parent with other index
    n_genes_from_other_parent = round((1 - bias) * n_genes)
    # Select which genes will be taken from parent with other index
    other_parent_genes = np.random.choice(
        gene_list, n_genes_from_other_parent, replace=False
    )
    # Replace corresponding values in each child chromosome
    child1_chromosome[other_parent_genes] = parent2_chromosome[
        other_parent_genes
    ]
    child2_chromosome[other_parent_genes] = parent1_chromosome[
        other_parent_genes
    ]
    # Initialize child object
    class_type = type(parent1)
    # Create child object1
    child1 = class_type(chromosome=child1_chromosome)
    child2 = class_type(chromosome=child2_chromosome)

    return child1, child2


def mutation(individual, mutation_rate, n_level, distance=-1):
    """
    Randomly choose genes for which to randomly choose a new value. The number
    of genes that will be changed is given by a mutation rate.
    How far can the new value can be, depends on the distance parameter.

    Parameters
    ----------
    individual : Stimuli
        The individual that is going to mutate.

    mutation_rate : float
        Probability of each gene to mutate.

    n_level : numpy array
        Number of different levels that each gene can take

    distance : int
        Defines the maximum level difference between the initial and mutated
        gene values.

    Returns
    -------
    mutated_individual : Stimuli
        Mutated individual.
    """
    # Maximum mutation distance
    # If not specified, set it above maximum distance in gene coding
    if distance < 0:
        distance = n_level.max() - 1
    # Only continue if maximum distance > 0
    if distance > 0:
        # Copy the initial individual before mutation
        mutated_individual_chromosome = individual.get_chromosome()
        # Get list of genes that can take more than one value
        gene_list = np.reshape(np.nonzero(n_level > 1), -1)
        # For each gene, mutate according to mutation rate
        for gene_idx in np.nditer(gene_list):
            prob = np.random.uniform()
            if prob < mutation_rate:
                new_gene_value = np.random.randint(n_level[gene_idx])
                while (
                    new_gene_value == mutated_individual_chromosome[gene_idx]
                    or abs(
                        new_gene_value
                        - mutated_individual_chromosome[gene_idx]
                    )
                    > distance
                ):
                    new_gene_value = np.random.randint(n_level[gene_idx])
                mutated_individual_chromosome[gene_idx] = new_gene_value
        # Initialize mutated individual object
        class_type = type(individual)
        # Create mutated individual object
        mutated_individual = class_type(
            chromosome=mutated_individual_chromosome
        )
    else:
        mutated_individual = individual.copy()
    return mutated_individual


def get_map_from_parametrization(parametrization):
    """
    Calculate a map between reparametrized values and original values before
    reparametrization, based on parametrization matrix.

    Parameters
    ----------
    parametrization : numpy array of boolean
        Matrix indicating which parameter levels of the full parametrization
        are valid.

    Returns
    -------
    map : list of numpy arrays
        Mapping to the original parametrization for each reparametrized gene
        value.
    """

    map_all = []
    for gene in range(parametrization.shape[0]):
        gene_map = np.reshape(np.nonzero(parametrization[gene, :]), -1)
        map_all.append(gene_map)
    return map_all
