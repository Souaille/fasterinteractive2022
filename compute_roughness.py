import librosa
import time
from joblib import Parallel, delayed
import os
from pathlib import Path
import numpy as np
from audio_features.utils import get_features_path
from mosqito.functions.roughness_danielweber import comp_roughness


def get_R_time_withsig(y_b, y_w, sr, cal, step, overlap=0.5):
    n_step = round(sr * step)
    n = y_w.shape[0]
    idx = 0
    Rmean_time = []
    while y_b[idx : (idx + n)].shape[0] == n:
        print("time: ", idx / sr)
        R = comp_roughness.comp_roughness(
            cal * (y_b[idx : (idx + n)] + y_w), sr, overlap
        )
        Rmean_time.append(np.mean(R["values"]))
        idx = idx + n_step
    return Rmean_time


def calc_roughness(file):
    # Print file name
    print(file)
    # Load background
    back_file_path = os.path.join("data", "audio_files", "projet1_background")
    back_file_name = "Beaubourg01_ORTF_42.wav"
    y_b, sr_b = librosa.load(
        os.path.join(back_file_path, back_file_name), sr=None
    )
    # Define source folder
    sound_dataset = "projet1_synth_spatmono"
    # Create full source folder path
    folder_src = os.path.join("data", "audio_files", sound_dataset)
    # Create destination file name
    filename_des = os.path.splitext(file)[0] + ".npy"
    # Create destination folders
    folder_des_mean = get_features_path("R_BSmean", sound_dataset, True)

    # Load file, upsample to match background sr
    y_w, _ = librosa.load(os.path.join(folder_src, file), sr=sr_b)

    # Compute mean roughness for signal + background at multiple times
    Rmean_time = get_R_time_withsig(
        y_b, y_w, sr_b, cal=12.01, step=2, overlap=0
    )
    # Average over all extracts
    Rmean_mean = np.mean(Rmean_time)

    # Create destination path if it does not already exist
    Path(folder_des_mean).mkdir(parents=True, exist_ok=True)
    # Save features
    np.save(
        os.path.join(folder_des_mean, filename_des), np.array([Rmean_mean])
    )


if __name__ == "__main__":
    # Define source folder
    sound_dataset = "projet1_synth_spatmono"
    # Create full source folder path
    folder_src = os.path.join("data", "audio_files", sound_dataset)
    file_list = os.listdir(folder_src)
    Parallel(n_jobs=4)(delayed(calc_roughness)(file) for file in file_list)
