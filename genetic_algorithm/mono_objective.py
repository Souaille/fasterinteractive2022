import numpy as np
import pandas as pd
import genetic_operators
import stimuli
import utils


class GABasic:
    """
    Class used to represent the IGA, but the subject is replaced by a model.
    Most of the code is adapted from Hadrien Marquez & Sébastien Villa's code.
    This version is a more classic version, where the genetic operators are
    replaced with more typical ones.


    Attributes
    ----------


    Methods
    -------



    """

    def __init__(
        self,
        parametrization,
        fitness_fun,
        pop_size=9,
        n_generation=11,
        mutation_rate=0.05,
        shuffle_pop=False,
        no_duplicates=False,
        crossover_bias=-1,
        max_mutation_distance=-1,
        elite_rate=0.2,
        fitness_noise_std=0,
        initialization="lhs",
        backup=False,
        initial_guess=None,
    ):
        self.parametrization = parametrization
        self.fitnessFun = fitness_fun
        self.popSize = pop_size
        self.nGeneration = n_generation
        self.mutationRate = mutation_rate
        self.geneSize = self.parametrization.shape[0]  # Might not be used
        self.nLevel = np.sum(self.parametrization, 1)
        self.shufflePopulation = shuffle_pop
        self.noDuplicates = no_duplicates
        self.crossoverBias = crossover_bias
        self.maxMutDist = max_mutation_distance
        self.nElites = round(self.popSize * elite_rate)
        self.fitnessNoiseStd = fitness_noise_std
        self.initialization = initialization
        self.initialGuess = utils.convert_guess(
            initial_guess, self.parametrization
        )

        self.doBackUp = backup

        self.generationIdx = 0
        # Creating map from reparametrized values to initial parameters
        self.map = genetic_operators.get_map_from_parametrization(
            self.parametrization
        )

        # Initialize data saving table columns
        data_backup_columns = (
            ["generation"]
            + ["p{}".format(i_gene) for i_gene in range(self.geneSize)]
            + ["score"]
        )
        # Preparing the data backup memory for one generation.
        self.dataBackupIGA = pd.DataFrame(
            0,
            index=np.arange(0, self.popSize * self.nGeneration),
            columns=data_backup_columns,
        )

        # Initializing arrays with metrics data
        self.metrics_mean = np.zeros(n_generation)
        self.metrics_max = np.zeros(n_generation)
        self.metrics_min = np.zeros(n_generation)

        # Initial population :
        self.population = genetic_operators.init_pop(
            self.popSize,
            self.nLevel,
            stimuli.Stimuli,
            initialization=self.initialization,
            initial_guess=self.initialGuess,
        )

        # Next population (created now just to have the right shape) :
        self.next_population = genetic_operators.init_pop(
            self.popSize,
            self.nLevel,
            stimuli.Stimuli,
        )

    def backup_data(self):
        last_param_name = "p" + str(self.geneSize - 1)

        # Backing up the chromosome
        for idx, stimulus in enumerate(self.population):
            row = idx + self.generationIdx * self.popSize
            self.dataBackupIGA.loc[
                row, "p0":last_param_name
            ] = stimulus.get_chromosome()
            self.dataBackupIGA.loc[row, "score"] = stimulus.get_score()
            self.dataBackupIGA.loc[row, "generation"] = self.generationIdx

    def update_metrics(self, fitness):
        # Update metrics based on current population data
        self.metrics_mean[self.generationIdx] = np.mean(fitness)
        self.metrics_max[self.generationIdx] = np.amax(fitness)
        self.metrics_min[self.generationIdx] = np.amin(fitness)

    def update_scores(self):
        # Update population with scores based on model data
        for idx, stimulus in enumerate(self.population):
            # Get encoded chromosome
            chromosome_encoded = np.copy(stimulus.get_chromosome())
            # Decode chromomose
            chromosome_decoded = utils.decode_param(
                chromosome_encoded, self.map
            )
            # Calculate score based on fitness function
            score = self.fitnessFun(chromosome_decoded)
            # print("Chrom: ", chromosome_decoded, ", fitness: ", score)
            # Save score to stimulus
            stimulus.set_score(score)

    def next_generation(self):
        # Generation to csv backup
        if self.doBackUp:
            self.backup_data()
        # Fitness calculation
        fitness = genetic_operators.calc_fitness(
            self.population, self.fitnessNoiseStd
        )
        # Update metrics
        self.update_metrics(fitness)

        # Initialize number of newly generated individuals
        idx_next_gen_indiv = 0
        # Select best individuals according to selection ratio and simply
        # copy them to the next generation
        fitness_order = np.argsort(fitness)
        # Maybe this should be improved
        # by priorizing recently rated solutions, if several solutions have
        # the same fitness. This could avoid unexpected behavior when users
        # rate using extreme integer values. #
        for idx in range(self.nElites):
            # self.next_population[idx] = self.population[
            #     fitness_order[-idx - 1]
            # ].copy()
            self.next_population[idx] = self.population[
                fitness_order[idx]
            ].copy()
            idx_next_gen_indiv += 1

        # For the remaining indivuals to create, apply genetic operators
        while idx_next_gen_indiv < self.popSize:

            # Select first parent with binary tournament
            parent1, parent1_idx = genetic_operators.binary_tournament(
                self.population, fitness
            )
            # Create mask so that the same parent cannot be selected
            # twice
            mask = np.arange(self.popSize) != parent1_idx
            # Select second parent with binary tournament
            parent2, _ = genetic_operators.binary_tournament(
                self.population[mask], fitness[mask]
            )
            # Perform crossover to create child
            child1, child2 = genetic_operators.uniform_crossover(
                parent1, parent2, self.crossoverBias
            )
            # Apply mutation on children
            child1_mut = genetic_operators.mutation(
                child1, self.mutationRate, self.nLevel, self.maxMutDist
            )
            child2_mut = genetic_operators.mutation(
                child2, self.mutationRate, self.nLevel, self.maxMutDist
            )

            # Copy child1 to next generation
            if (
                self.noDuplicates
                and (
                    child1_mut not in self.next_population[:idx_next_gen_indiv]
                )
            ) or (not self.noDuplicates):
                self.next_population[idx_next_gen_indiv] = child1_mut
                # Increment individual counter
                idx_next_gen_indiv += 1

            # Copy child2 to next generation if still space
            if idx_next_gen_indiv < self.popSize:
                if (
                    self.noDuplicates
                    and (
                        child1_mut
                        not in self.next_population[:idx_next_gen_indiv]
                    )
                ) or (not self.noDuplicates):
                    self.next_population[idx_next_gen_indiv] = child2_mut
                    # Increment individual counter
                    idx_next_gen_indiv += 1
        # Shuffle population order to randomize evaluation order if desired
        if self.shufflePopulation:
            self.next_population = np.random.shuffle(self.next_population)
        # Individually copy each individual into current population
        for i in range(self.popSize):
            self.population[i] = self.next_population[i].copy()

    def run(self):
        for generation_idx in range(self.nGeneration):
            self.generationIdx = generation_idx
            # print("Gen: ", generation_idx)
            self.update_scores()
            self.next_generation()

    def get_backup(self):
        return self.dataBackupIGA

    def get_metrics(self):
        # Calculate and return metrics arrays
        return self.metrics_max, self.metrics_mean, self.metrics_min
