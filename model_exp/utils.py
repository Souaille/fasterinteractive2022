import os
import numpy as np


def get_model_coefs(model, idx=-1):
    """
    This function loads the model coefficients calculated on a given dataset,
    with or without interaction terms.

    Parameters
    ----------
    model: string


    Returns
    -------
    param_val_list
        A numpy array where each row represents the levels of the input
        parameters corresponding to a given model coefficient.
    """

    path = get_path(model, idx)
    try:
        coef = np.genfromtxt(path, delimiter=";")
    except FileNotFoundError:
        print("Model file does not exist at: ", path)
        coef = np.array([])

    return coef


def get_path(model, idx=-1):
    folder = get_folder()
    file = get_file_name(model, idx)
    path = os.path.join(folder, file)
    return path


def get_folder():
    folder = os.path.join("data", "perceptual_models")
    return folder


def get_file_name(model, idx=-1):

    dictionnary = {
        "UnpLin": "unpleasantness_lin",
        "DetLin": "detectability_lin",
        "UnpLinExc": "unpleasantness_lin_excl",
        "DetLinExc": "detectability_lin_excl",
    }
    if idx > -1:
        tag = str(idx)
    else:
        tag = ""

    model_file = dictionnary[model] + tag + ".csv"
    return model_file
