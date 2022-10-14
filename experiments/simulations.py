import mono_objective as mono_GA
from model_exp.model import get_model
import numpy as np
import pandas as pd
import os
from itertools import combinations, product
from audio_features.features import Features
import scipy.stats as st
from tqdm import tqdm


def run_IGA(
    reparam_mat,
    model_fun,
    backup=False,
    initial_guess=None,
    initialization="lhs",
):
    """
    Define IGA simulation run
    """
    IGA = mono_GA.GABasic(
        reparam_mat,
        model_fun,
        pop_size=9,
        n_generation=11,
        mutation_rate=0.05,
        shuffle_pop=False,
        no_duplicates=False,
        crossover_bias=-1,
        max_mutation_distance=-1,
        elite_rate=0.2,
        fitness_noise_std=0,
        initialization=initialization,
        backup=backup,
        initial_guess=initial_guess,
    )
    IGA.run()
    (max_fit, mean_fit, min_fit) = IGA.get_metrics()
    if backup:
        backup = IGA.get_backup()
        return max_fit, mean_fit, min_fit, backup
    else:
        return max_fit, mean_fit, min_fit


def update_param_mat(param_mat, idxs):
    param_mat_c = param_mat.copy()
    param_mat_c[:, :] = 0
    for row, idx in enumerate(idxs):
        param_mat_c[row, idx] = 1

    return param_mat_c


def get_kidxs_mean(
    lvl_list, x_sample, y_sample, n_tokeep, minmax="min", get_vals=False
):
    """
    Define function that returns the reparametered matrix based on optimal location
    """
    # Get indices of levels to keep based on means of data sample
    idxs = []
    mean_vals = []
    sem_vals = []
    for param_id, lvls in enumerate(lvl_list):
        mean_param = []
        sem_param = []
        for lvl in lvls:
            mask = x_sample[:, param_id] == lvl
            data = np.squeeze(y_sample[mask])

            # Compute mean
            mean = np.mean(data)
            mean_param.append(mean)

            sem = st.sem(data)
            sem_param.append(sem)

        # Keep desired number of levels that minimize/maximize the mean
        if minmax == "min":
            idx = np.argsort(mean_param)[:n_tokeep]
        elif minmax == "max":
            idx = np.argsort(mean_param)[-n_tokeep:]
        #         print(idx)
        idxs.append(idx)
        mean_vals.append(mean_param)
        sem_vals.append(sem_param)
    if get_vals:
        return idxs, np.asarray(mean_vals), np.asarray(sem_vals)
    else:
        return idxs


def run_full_tests_no_interp_mean(
    nb_models,
    nb_runs,
    n_tokeep_list,
    method,
    backup=False,
    initial_guess=None,
    model="UnpLin",
    features=None,
    minmax="min",
    selection="paramwise",
):


    lvl_list = [
        np.arange(4),
        np.arange(4),
        np.arange(4),
        np.arange(4),
        np.arange(4),
        np.arange(4),
    ]
    param_mat = np.ones((6, 4), dtype=bool)
    full_mat = np.array(list(product(range(4), repeat=6)))
    sound_dataset = "projet1_synth_spatmono"
    data_mean = {}
    data_min = {}
    for n_tokeep in n_tokeep_list:

        if features is None and n_tokeep != 4:
            continue

        if features is not None and n_tokeep == 4:
            continue
        print("N to keep: ", n_tokeep)
        mmin_fit = []
        mmean_fit = []
        for mdl_idx in tqdm(range(nb_models)):
            model_fun = get_model(model, mdl_idx + 1, cont=False)
            mean_fit_list = []
            min_fit_list = []

            if features is None and n_tokeep == 4:
                idxs = [
                    np.arange(4),
                    np.arange(4),
                    np.arange(4),
                    np.arange(4),
                    np.arange(4),
                    np.arange(4),
                ]
            elif ("Unp" in features) or ("Det" in features):
                features_fun = get_model(features, mdl_idx + 1, cont=False)
                full_fact_val = []
                for param in full_mat:
                    full_fact_val.append(features_fun(param + 1))
                full_features = np.array(full_fact_val)
                if selection == "paramwise":
                    idxs = get_kidxs_mean(
                        lvl_list,
                        full_mat,
                        full_features,
                        n_tokeep,
                        minmax=minmax,
                    )
                
            else:
                features_obj = Features(
                    sound_dataset=sound_dataset,
                    feature_set=features,
                    time_average=True,
                )
                full_features = features_obj.get_features(full_mat)
                if selection == "paramwise":
                    idxs = get_kidxs_mean(
                        lvl_list,
                        full_mat,
                        full_features,
                        n_tokeep,
                        minmax=minmax,
                    )

            reparam_mat = update_param_mat(param_mat, idxs)

            for idx in range(nb_runs):
                (_, mean_fit, min_fit) = run_IGA(
                    reparam_mat,
                    model_fun,
                    backup=backup,
                    initial_guess=initial_guess,
                )
                mean_fit_list.append(mean_fit)
                min_fit_list.append(min_fit)

            mmin_fit = mmin_fit + min_fit_list
            mmean_fit = mmean_fit + mean_fit_list
        key = method + " " + str(n_tokeep) + " levels"
        data_mean[key] = mmean_fit
        data_min[key] = mmin_fit

    return data_min, data_mean


def run_fifty_fifty_mean(
    sim_model,
    n_tokeep_list,
    nb_runs,
    features_list,
    name_list,
    minmax_list,
    selection="paramwise",
):
    nb_models = 28
    data_min, data_mean = {}, {}
    # Make simulations
    for features, name, minmax in zip(features_list, name_list, minmax_list):
        print("Features :", features)
        data_min_temp, data_mean_temp = run_full_tests_no_interp_mean(
            nb_models,
            nb_runs,
            n_tokeep_list,
            name,
            backup=False,
            initial_guess=None,
            model=sim_model,
            features=features,
            minmax=minmax,
            selection=selection,
        )

        data_min.update(data_min_temp)
        data_mean.update(data_mean_temp)

    return data_min, data_mean
