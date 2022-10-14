import numpy as np
import matplotlib.pyplot as plt


def convert_guess(guess_pop, parametrization):
    if guess_pop is not None:
        conv_guess = np.copy(guess_pop)
        rev_map = get_rev_map(parametrization)

        for idx in range(guess_pop.shape[0]):
            for idx2, val in enumerate(guess_pop[idx, :]):
                accepted_vals = np.reshape(
                    np.where(parametrization[idx2, :]), -1
                )
                if not np.any(accepted_vals == val):
                    raise ValueError("Guess value outside parametrization.")
            conv_guess[idx, :] = encode_param(guess_pop[idx, :], rev_map)
    else:
        conv_guess = np.array([])
    return conv_guess


def get_rev_map(parametrization):
    rev_map = np.ones_like(parametrization, dtype=int) * -1
    for idx in range(parametrization.shape[0]):
        cnt = 0
        for idx2, val in enumerate(parametrization[idx, :]):
            if val:
                rev_map[idx, idx2] = cnt
                cnt = cnt + 1
    return rev_map


def encode_param(chromosome_decoded, rev_map):
    chromosome_encoded = np.copy(chromosome_decoded)
    for idx, gene in enumerate(chromosome_encoded):
        chromosome_encoded[idx] = rev_map[idx][gene]
    return chromosome_encoded


def decode_param(chromosome_encoded, map):
    chromosome_decoded = np.copy(chromosome_encoded)
    for idx, gene in enumerate(chromosome_decoded):
        chromosome_decoded[idx] = map[idx][gene]
    chromosome_decoded += 1  # 1 has to be added
    # because min param level in list starts at 1, while it starts
    # 0 in IGA code. This should be homogeneized later.
    return chromosome_decoded


def calculate_duration_score(timerValue, tStart, stimuliDuration):
    """
    Calculate the duration score: dScore.

    The duration score is given by timerValue - tStart.
    If the timerValue is inferior to the start or superior to the end time of
    the sample, the timerValue is changed to an average value.

    Parameters
    ----------
    timerValue : float
        The user detection time recorded.

    tStart : float
        The random start of the stimuli sample.

    stimuliDuration : int
        The duration in second of the stimuli sample.

    Returns
    -------
    float
        The duration score.
    """
    if tStart + stimuliDuration <= timerValue < tStart:
        timerValue = tStart + stimuliDuration / 2
    return timerValue - tStart


def plot_best_fitness_progress(fitness, ax=None):
    """
    Plots the best fitness value of each generation (iteration of the IGA).

    Parameters
    ----------
    fitness : numpy array
        The fitness value in a numpy array.
        The number of rows is equal to the population size.
        The number of columns is equal to the number of generation.

    Returns
    -------

    """
    # Finding best candidate in each generation
    best = np.array([generation.max() for generation in fitness.transpose()])

    # Ploting the data.
    ax = plt.gca()
    ax.set_title("Best fitness score per generation")
    ax.set_xlabel("Generation number")
    ax.set_ylabel("Fitness score")
    return ax.plot(best, "o")
