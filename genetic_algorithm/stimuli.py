import numpy as np
from genetic_algorithm import utils


class Stimuli:
    """
    Base class used to represent the stimuli associated to a chromosome of the
    population.

    Custom made stimuli must inherit from this base class to work with IGA
    implementation.

    Attributes
    ----------

    score : float
        The score given by the user for this stimuli.

    chromosome : numpy array
        Contains the genes of the individual associated to the stimuli.
        The genes are binary encoded.

    Methods
    -------
     set_score(score)
         Sets the uScore attribute.

     set_chromosome(chromosome)
         Sets the chromosome attribute.
         Chromosome must be given in binary form.

     get_score()
         Returns the dScore attribute.

     get_chromosome()
         Returns the chromosome attribute.
         The chromosome is given in binary form.

    """

    def __init__(self, nLevel=0, chromosome=None):
        """
        Parameters
        ----------
        nLevel : list
            The number of possible values for a gene.
        chromosome : numpy array
            A chromosome. If passed the stimuli will copy this chromosome.
            If not it will generate a random chromosome based on nLevel,
            geneSize and nBits.
        """

        if chromosome is None:
            geneSize = nLevel.shape[0]
            self.chromosome = np.array(
                [np.random.randint(nLevel[i]) for i in range(geneSize)]
            )
        else:
            self.chromosome = np.copy(chromosome)

    def set_score(self, score):
        self.score = score

    def set_chromosome(self, chromosome):
        self.chromosome = np.copy(chromosome)

    def set_fitness(self, fitness):
        self.fitness = np.copy(fitness)

    def set_p2_rank(self, p2_rank):
        self.p2_rank = np.copy(p2_rank)

    def get_score(self):
        return self.score

    def get_chromosome(self):
        return np.copy(self.chromosome)

    def get_fitness(self):
        return np.copy(self.fitness)

    def get_p2_rank(self):
        return np.copy(self.p2_rank)

    def copy(self):
        new_stim = Stimuli(chromosome=self.get_chromosome())
        return new_stim

    def __repr__(self):
        return "Chromosome with " + str(self.chromosome.shape[0]) + " genes."
