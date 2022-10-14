import os
import csv
import numpy as np
from itertools import product


def get_all_files_list(feature_set, sound_dataset, time_average):
    # Define path to files
    src_folder = get_features_path(feature_set, sound_dataset, time_average)
    # Get list of all content of folder
    file_list = [os.path.join(src_folder, f) for f in os.listdir(src_folder)]
    # Only keep numpy files
    file_list = [f for f in file_list if f.endswith(".npy")]
    # Extract sound names
    if sound_dataset in [
        "projet1_synth",
        "exp3bis",
        "projet1_synth_spatmono",
        "projet1_synth_no_fade",
    ]:
        sound_names = [
            os.path.splitext(os.path.basename(path))[0].split("_")[0]
            for path in file_list
        ]
    else:
        sound_names = [
            os.path.splitext(os.path.basename(path))[0] for path in file_list
        ]
    return file_list, sound_names


def get_time_av_folder(time_average):
    """
    Generate folder name to use when dealing with time averaged data.

    Parameters
    ----------
    time_average : boolean
        Description of parameter `time_average`.

    Returns
    -------
    string
        Folder name.

    """
    # Define time average subfolder
    if time_average:
        time_av_folder = "time_averaged"
    else:
        time_av_folder = "not_averaged"
    return time_av_folder


def get_features_path(feature_set, sound_dataset, time_average):
    """
    Creates path for saving audio features.

    Parameters
    ----------
    feature_set : string
        Feature set name
    sound_dataset : string
        Sound dataset.
    time_average : boolean
        Whether the features are averaged over time or not.

    Returns
    -------
    features_path
        Path to feature files.

    """

    time_av_folder = get_time_av_folder(time_average)
    features_path = os.path.join(
        "data", "features", feature_set, sound_dataset, time_av_folder
    )
    return features_path


def get_sound_name_from_param(sound_param):
    """
    This function returns the sound name corresponding to the input parameters.
    Parameters
    ----------
        sound_param : numpy array of int
            List of parameters that define the sound.
    Returns
    -------
        sound_name : string
            Name of the sound.
    """

    sound_param_corr = sound_param.copy()
    # Correct for parametrization redundancy
    if sound_param.ndim == 1:
        if sound_param_corr[2] == 0:
            sound_param_corr[0] = 0
            sound_param_corr[3] = 0
        if sound_param_corr[5] == 0:
            sound_param_corr[4] = 0
        sound_name = "".join(map(str, sound_param_corr.tolist()))
    elif sound_param.ndim == 2:
        sound_name = []
        for param in sound_param_corr:
            if param[2] == 0:
                param[0] = 0
                param[3] = 0
            if param[5] == 0:
                param[4] = 0

            sound_name.append("".join(map(str, param.tolist())))
    return sound_name


def correct_sound_name(sound_name):
    new_name = sound_name
    if new_name[2] == "0":
        new_name = "0" + new_name[1] + "00" + new_name[4:]
    if new_name[5] == "0":
        new_name = new_name[:4] + "00"

    return new_name