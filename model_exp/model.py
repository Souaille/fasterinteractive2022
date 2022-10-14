"""
This module defines perceptual models.
"""

import numpy as np
from SALib.test_functions import Sobol_G
from model_exp.utils import get_model_coefs


def get_model(model, idx=-1, cont=False, scales=None):
    """
    Return model function.

    Parameters
    ----------
    model : string
        Model name

    idx : int
        In the case of a perceptual model, selects the model version.
        In the case of a test model, effect depends on the model.
    cont : boolean
        Set to True if model values should be interpolated between discrete
        values in initial model.

    Returns
    -------
    model_predict : function
        Model predictor function.

    """
    perceptual_models = [
        "UnpLin",
        "DetLin",
        "UnpLinExc",
        "DetLinExc",
    ]
    if model in perceptual_models:

        def model_predict(param, index=0):
            fun = predict_perceptual_model(param, model, idx, cont, scales)
            return fun


    return model_predict


def predict_perceptual_model(param, model, idx=-1, cont=False, scales=None):
    """
    This function predicts values given by a perceptual model fitted on data.

    Parameters
    ----------
    param: numpy array
        Parameter values
    model : string
        Name of model to use.

    Returns
    -------
    value : float
        Calculated model value.

    """
    coefs = get_model_coefs(model, idx)
    n_param = 6
    n_levels = 4
    if "Int" in model:
        interaction = True
    else:
        interaction = False
    if cont:
        value = predict_model_lin_cont(param, n_param, n_levels, coefs, scales)
    else:
        value = predict_model_lin(param, n_param, n_levels, interaction, coefs)
    return value


def predict_model_lin(param, n_param, n_levels, interaction, coefs):
    """
    This function returns the value predicted by a linear model.

    Parameters
    ----------
    param: numpy array
        Parameter values
    n_param : int
        Number of parameters
    n_levels : int
        Number of levels per parameter.
    interaction : boolean
        Whether or not to consider interactions.
    coefs : numpy array
        Model coefficients.

    Returns
    -------
    value : float
        Calculated model value.
    """
    if np.max(param) > n_levels:
        print("Input value outside model range.")
        return np.nan
    param_val_list = get_param_val_list(n_param, n_levels, interaction)
    value = coefs[0]
    coefs = coefs[1:]
    if interaction:
        for p_1 in range(n_param):
            idx1 = param_val_list[:, p_1] == param[p_1]
            new_val = coefs[
                np.logical_and(
                    idx1,
                    np.count_nonzero(param_val_list, axis=1) == 1,
                )
            ]
            value += new_val.item()
            for p_2 in range(p_1 + 1, n_param):
                idx2 = param_val_list[:, p_2] == param[p_2]
                new_val = coefs[np.logical_and(idx1, idx2)]
                value += new_val.item()
    else:
        for p in range(n_param):
            idx = param_val_list[:, p] == param[p]
            new_val = coefs[idx]
            value += new_val.item()
    return value


def predict_model_lin_cont(param, n_param, n_levels, coefs, scales=None):
    """
    This function returns the value predicted by a linear model and
    performs linear interpolation for values in between the initial levels.
    This model does not work for models with interactions.

    Parameters
    ----------
    param: numpy array
        Parameter values
    n_param : int
        Number of parameters
    n_levels : int
        Number of levels per parameter in initial model.
    coefs : numpy array
        Model coefficients.
    scales : numpy array or None
        Number of levels after interpolation. Used to scale value to initial
        intervals.
    Returns
    -------
    value : float
        Calculated model value.
    """
    if scales is not None:
        param = (param - 1) / (scales - 1) * (n_levels - 1) + 1

    if np.max(param) > n_levels:
        print("Input value outside model range.")
        return np.nan
    param_val_list = get_param_val_list(n_param, n_levels, False)
    value = coefs[0]
    coefs = coefs[1:]

    for p in range(n_param):
        idx_f = param_val_list[:, p] == np.floor(param[p])
        idx_c = param_val_list[:, p] == np.ceil(param[p])

        val_f = coefs[idx_f].item()
        val_c = coefs[idx_c].item()
        if np.argwhere(idx_f).item() == np.argwhere(idx_c).item():
            new_val = val_f
        else:
            new_val = val_f + (val_c - val_f) * (
                param[p] - np.floor(param[p])
            ) / (np.ceil(param[p]) - np.floor(param[p]))

        value += new_val
    return value


def get_param_val_list(n_param, n_levels, interaction=False):
    """
    This function creates a list of the parameter levels corresponding to each
    coefficient in the model.

    Parameters
    ----------
    n_param : int
        The number of parameters to consider.

    n_levels : int
        The number of levels that each parameter can take.

    interaction : bool
        Whether or not to consider interactions between parameters for the
        model.

    Returns
    -------
    param_val_list
        A numpy array where each row represents the levels of the input
        parameters corresponding to a given model coefficient.
    """

    # Create array with list of parameter values for linear model coefficients
    lin_val_list = np.zeros((n_param * n_levels, n_param), dtype=int)
    for param in range(n_param):
        for level in range(n_levels):
            lin_val_list[param * n_levels + level, param] = level + 1

    param_val_list = lin_val_list

    # Create array with list of parameter values for interaction model
    # coefficients
    if interaction:
        int_val_list = np.zeros(
            (np.sum(np.arange(n_param)) * n_levels ** 2, n_param), dtype=int
        )
        idx = 0
        for param1 in range(n_param - 1):
            for param2 in range(param1 + 1, n_param):
                for level1 in range(n_levels):
                    for level2 in range(n_levels):
                        int_val_list[idx, param1] = level1 + 1
                        int_val_list[idx, param2] = level2 + 1
                        idx += 1
        param_val_list = np.append(param_val_list, int_val_list, axis=0)
    return param_val_list
