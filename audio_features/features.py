import os
import librosa
import numpy as np
import pickle
from tqdm import tqdm
from audio_features.utils import (
    get_all_files_list,
    get_sound_name_from_param,
    get_features_path,
    correct_sound_name,
)
import timbral_models
import time


class Features:
    def __init__(self, sound_dataset, feature_set, time_average):
        self.sound_dataset = sound_dataset
        self.feature_set = feature_set
        self.time_average = time_average
        _, self.sound_names = get_all_files_list(
            feature_set, sound_dataset, time_average
        )
        self.features_mat = load_set_to_matrix(
            [], sound_dataset, feature_set, time_average
        )
        self.fixed_values = None

    def set_fixed_values(self, values):
        self.fixed_values = values.copy()

    def get_features(self, sound_param):
        if self.fixed_values is not None:
            for pair in self.fixed_values:
                sound_param = np.insert(sound_param, pair[0], pair[1], axis=1)
        # Get corresponding sound names
        sound_names = get_sound_name_from_param(sound_param)
        # Get corresponding indices from full list
        sound_id = [self.sound_names.index(name) for name in sound_names]
        # Get corresponding features
        features_mat = self.features_mat[sound_id, :]
        return features_mat

def load_set_to_matrix(
    sound_names, sound_dataset, feature_set, time_average, verbose=False
):
    """
    Load features from dataset and arrange as a matrix sound*feature

    Parameters
    ----------
    sound_name : list of strings
        List of sound names.
    sound_dataset : string
        Dataset the sound belongs to.
    feature_set : string
        Commputed features set.
    time_average : boolean
        Whether the features should be averaged over time.

    Returns
    -------
    numpy array
        Features matrix

    """
    # Initialize matrix
    features_mat = np.array([])
    # Only compute if features not None, otherwise return empty array
    if feature_set != "none":
        # Get full dictionnary
        full_dict = get_full_dict(sound_dataset, feature_set, time_average)
        # If no sound names are specified, get sound names from dict keys
        if len(sound_names) == 0:
            dict = full_dict
        # If synth1 is used, add required keys to correct dictionnary
        elif sound_dataset in [
            "projet1_synth",
            "exp3bis",
        ]:
            dict = {
                key: full_dict[correct_sound_name(key)] for key in sound_names
            }
        # Else only keep required keys
        else:
            # print(full_dict.keys())
            dict = {key: full_dict[key] for key in sound_names}
        # Loop through sounds
        if verbose:
            print("Building matrix")
            print(len(dict))
        for key in dict.keys():
            # Load features
            features_arr = dict[key]
            # Reshape to row
            features_row = reshape_row(features_arr)
            # Concatenate with previous arrays
            if features_mat.size:
                features_mat = np.concatenate(
                    (features_mat, features_row), axis=0
                )
            else:
                features_mat = features_row.copy()
    return features_mat


def get_full_dict(sound_dataset, feature_set, time_average, verbose=False):

    feat_path = get_features_path(feature_set, sound_dataset, time_average)
    # Try loading the dictionnary
    try:
        with open(os.path.join(feat_path, "full_dict.pkl"), "rb") as infile:
            full_dict = pickle.load(infile)
            if verbose:
                print("Loaded dictionnary.")
    # Compute and save if it does not work
    except:
        if verbose:
            print("Computing dictionnary.")
        # Get list of all feature files
        file_list, sound_names = get_all_files_list(
            feature_set, sound_dataset, time_average
        )
        # Initialize feature list
        features_list = []
        # Loop over files and append list
        for sound_name in sound_names:
            features_arr = load_from_file(
                sound_name, sound_dataset, feature_set, time_average
            )
            # If any of the arrays is empty, return an empty dictionnary and do
            # not save it.
            if features_arr.size == 0:
                return {}
            features_list.append(features_arr)
        # Create dictionnary from sound names and feature arrays
        full_dict = {
            sound_name: features
            for (sound_name, features) in zip(sound_names, features_list)
        }
        # Save dictionnary if features were computed
        with open(os.path.join(feat_path, "full_dict.pkl"), "wb") as outfile:
            pickle.dump(full_dict, outfile, pickle.HIGHEST_PROTOCOL)
    return full_dict


def load_from_file(sound_name, sound_dataset, feature_set, time_average=False):
    """
    Load features from precommpted .npy file based on sound name.

    Parameters
    ----------
    sound_name : string
        Sound name.
    sound_dataset : string
        Dataset the sound belongs to.
    feature_set : string
        Commputed features set.
    time_average : boolean
        Whether the features should be averaged over time.

    Returns
    -------
    numpy array
        Features.

    """
    # Some datasets only have time averaged data, take that into account
    only_av = ["gene_Ma18"]
    if sound_dataset in only_av:
        time_average = True
    # Get list of all files corresponding to required input
    file_list, sound_names = get_all_files_list(
        feature_set, sound_dataset, time_average
    )
    # Find file corresponding to sound
    # file_names = [
    #     s
    #     for s in file_list
    #     if os.path.splitext(sound_name)[0]
    #     in os.path.splitext(os.path.basename(s))[0]
    # ]
    # print(sound_names)
    # print(file_list)
    file_names = [
        fn for (sn, fn) in zip(sound_names, file_list) if sound_name == sn
    ]
    # Print warning and return empty array if not unique file found
    if len(file_names) == 0:
        print("No file found for ", sound_name, "\n")
        return np.array([])
    if len(file_names) > 1:
        print("More than one feature file found for name: ", sound_name)
        return np.array([])
    # Load features file
    features_arr = np.load(os.path.join(file_names[0]))
    return features_arr


def reshape_row(features_arr):
    """
    Reshape features array as a row vector.

    Parameters
    ----------
    features_arr : numpy array
        Features array.


    Returns
    -------
    numpy array
        Reshaped array.

    """
    features_row = np.reshape(features_arr, (1, -1))
    return features_row

