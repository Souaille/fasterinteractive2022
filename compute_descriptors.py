#!/usr/bin/env python
# coding: utf-8

# This notebook aims to find and compute relevant loudness related indicators

# # I. Module imports

# In[1]:

import librosa
import time
from joblib import Parallel, delayed
import os
from pathlib import Path
import numpy as np
from audio_features.utils import get_features_path
from mosqito.functions.roughness_danielweber import comp_roughness


import os
import librosa
import timbral_models
import numpy as np
from tqdm import tqdm


# In[2]:


import librosa.display


# In[3]:


import scipy.stats


# In[4]:


# import matplotlib.pyplot as plt
# from matplotlib.ticker import FormatStrFormatter
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['figure.dpi'] = 200


# In[5]:


from pathlib import Path


# In[6]:


from audio_features.features import Features
from audio_features.utils import get_features_path


# In[7]:


import pandas as pd


# In[8]:


# get_ipython().run_line_magic('load_ext', 'autoreload')


# In[9]:


# get_ipython().run_line_magic('autoreload', '2')


# In[10]:


from itertools import combinations, product


# In[11]:


import time


# In[12]:


from mosqito.functions.loudness_zwicker import comp_loudness


# In[13]:


from mosqito.functions.oct3filter.oct3spec import oct3spec
from mosqito.functions.oct3filter.comp_third_spectrum import comp_third_spec
from mosqito.functions.oct3filter.calc_third_octave_levels import calc_third_octave_levels
from mosqito.functions.loudness_zwicker.loudness_zwicker_stationary import loudness_zwicker_stationary


# ### dBA from power spectrum

# In[14]:


def L_to_dBA(F, P):
    # Compute A weights with librosa
    Aw = librosa.A_weighting(F)
    # Compute A-weighted levels
    Pw = P + Aw
    # Compute total A-weighted level
    Pwtot = 10*np.log10(np.sum(10**(Pw/10)))
    return Pwtot


# ### dBA from signal

# In[15]:


def sig_to_dBA(y, Pref, sr, Fmin, Fmax, N):
    _, P, F = timbral_models.filter_third_octaves_downsample(y, Pref, sr, Fmin, Fmax, N)
    PA = L_to_dBA(F, P)
    return PA


# ### Mean dN over several background extracts

# In[16]:


def get_N_change_wb(y_b, y_w, sr, Pref, t_event=5.4, step=2):

    dt = 1.5
    w_length = int(round(0.5*sr))
    idx_event = int(round(t_event*sr))
    idx_prec = int(round((t_event-dt)*sr))
    
    n = y_w.shape[0]
    dN_time = []
    idx = 0
    n_step = round(step*sr)
    while y_b[idx:(idx+n)].shape[0] == n:
        # Mix warning signal with background for each window
        y = y_b[idx:(idx+n)] + y_w
        # Compute loudness for the two windows preceeding detection
        N_1, _ = timbral_models.specific_loudness(y[(idx_prec-w_length):idx_prec], Pref, sr, 0)
        N_2, _ = timbral_models.specific_loudness(y[(idx_event-w_length):idx_event], Pref, sr, 0)
        # Compute loudness difference
        dN = N_2 - N_1
        dN_time.append(dN)
        idx = idx + n_step 
    dN_av = np.mean(dN_time)
    return dN_av


# ### Get list of background extract mean power spectrums 

# In[17]:


def get_F_back_extracts(y_b, sr, n_war, step=2, freq_scale="lin", n_fft=2048):
    S_back_list = []
    
    idx = 0
    n_step = round(step*sr)
    while y_b[idx:(idx+n_war)].shape[0] == n_war:
        y_1 = y_b[idx:(idx+n_war)]
        
        if freq_scale == "lin":  
            S1 = np.sqrt(np.abs(librosa.stft(y_1, n_fft=n_fft)**2).mean(axis=1))
        elif freq_scale == "mel":
            S1 = np.sqrt(librosa.feature.melspectrogram(y_1, sr=sr, n_fft=n_fft).mean(axis=1))
            

        
        S_back_list.append(S1)
        
        idx = idx + n_step
        
    return S_back_list


# ### Compute difference for a given warning signal and average over given backround loudness profiles

# In[18]:


def get_F_diff_wb_full(S_back_list, y_w, sr, method="MAE", only_pos=False, freq_scale="lin", amp_scale="lin", n_fft=2048):

    dF_time = []

    
    for S1 in S_back_list:
        y_2 = y_w
        
        if freq_scale == "lin":  
            S2 = np.sqrt(np.abs(librosa.stft(y_2, n_fft=n_fft)**2).mean(axis=1))
        elif freq_scale == "mel":
            S2 = np.sqrt(librosa.feature.melspectrogram(y_2, sr=sr, n_fft=n_fft).mean(axis=1))
        else:
            print("Frequency scale [", freq_scale, "] is not an option")
        
        if amp_scale == "lin":
            diff = S2 - S1
        elif amp_scale == "log":
            S1_dB = librosa.amplitude_to_db(S1)
            S2_dB = librosa.amplitude_to_db(S2)
            diff = S2_dB - S1_dB
        elif amp_scale == "pow":
            diff = S2**2 - S1**2
        else:
            print("Amplitude scale [", amp_scale, "] is not an option")
            
        
        if only_pos:
            diff[diff<0] = 0
        if method == "MAE":
            dF = np.mean(np.abs(diff))
        elif method == "MSE":
            dF = np.mean(np.abs(diff)**2)
        elif method == 'logsum':
            dF = 10*np.log10(np.sum(10**(np.abs(diff)/10))/diff.shape[0])
        dF_time.append(dF)
        
    dF_av = np.mean(dF_time)
    return dF_av


# ### Loudness over background + warning with a moving window, with Timbral Models

# In[19]:


def get_N_time_withsig(y_b, y_w, sr, Pref, step, cal_back=1):
    n = y_w.shape[0]
    idx = 0
    N_time = []
    while y_b[idx:(idx+n)].shape[0] == n:
        N_t, _ = timbral_models.specific_loudness(cal_back*(y_b[idx:(idx+n)] + y_w), Pref, sr, 0)
        N_time.append(N_t)
        idx = idx + step                
    return N_time


# ## Define the path to the background sound file and import it

# In[20]:


file_path = os.path.join("data", "audio_files", "projet1_background")
file_name = "Beaubourg01_ORTF_42.wav"


# In[21]:


y_b, sr_b = librosa.load(os.path.join(file_path, file_name), sr=None)


# ## Find reference level to match background level target

# Compute level in third octave bands and corresponding center frequencies

# In[22]:


Pref = -20*np.log10(20e-6) # This is the reference value that should be used to get dB SPL from Pa signal.
Fmin = 25
Fmax = 12500
N = 4


# Get power per 1/3 octave bands

# In[23]:


_, P, F = timbral_models.filter_third_octaves_downsample(y_b, Pref, sr_b, Fmin, Fmax, N)


# Get total power level in dBA

# In[24]:


Pw_tot = sig_to_dBA(y_b, Pref, sr_b, Fmin, Fmax, N)
print('Level: ', Pw_tot, 'dBA')


# Get calibration factor based on target level, which is 69dBA

# In[25]:


Pw_target = 69
cal_back = 10**((Pw_target- Pw_tot)/20)


# # Compute descriptors

# ## N_BS

# In[26]:


# Define source folder
sound_dataset = 'projet1_synth_spatmono'
# Create full source folde path
folder_src = os.path.join('data', 'audio_files', sound_dataset)
# Create destination path based on naming convention for the projet
features = 'N_BS'
folder_des = get_features_path(features, sound_dataset, True)


# In[27]:


# Define time step between each window
d_step = 2
step = round(sr_b*d_step)
for file in tqdm(os.listdir(folder_src)):
    # Create destination file name
    filename_des = os.path.splitext(file)[0] + '.npy'
    # Check if features already computed for this file
    deja_vu = os.path.exists(os.path.join(folder_des, filename_des))
    # Compute if not already done
    if not deja_vu:
        # Load file, upsample to match background sr
        y_w, sr_w = librosa.load(os.path.join(folder_src, file), sr=sr_b)
        # Compute loudness for signal combined with different background sections
        N_arr = get_N_time_withsig(cal_back*y_b, cal_back*y_w, sr_b, Pref, step)
        # Average
        N_av = np.mean(N_arr, keepdims=True)
        # Create destination path if it does not already exist
        Path(folder_des).mkdir(parents=True, exist_ok=True)
        # Save features
        np.save(os.path.join(folder_des, filename_des), N_av)


# ## dNpeak_BS

# Compute loudness difference, with peak time as reference, with background.

# In[28]:


# Define source folder
sound_dataset = 'projet1_synth_spatmono'
# Create full source folde path
folder_src = os.path.join('data', 'audio_files', sound_dataset)
# Create destination path based on naming convention for the projet
features = 'dNpeak_BS'
folder_des = get_features_path(features, sound_dataset, True)


# In[29]:


for file in tqdm(os.listdir(folder_src)):
    # Create destination file name
    filename_des = os.path.splitext(file)[0] + '.npy'
    # Check if features already computed for this file
    deja_vu = os.path.exists(os.path.join(folder_des, filename_des))
    # Compute if not already done
    if not deja_vu:
        # Load file, upsample to match background sr
        y_w, sr_w = librosa.load(os.path.join(folder_src, file), sr=sr_b)
        # Compute loudness for signal alone
        dN = get_N_change_wb(cal_back*y_b, cal_back*y_w, sr_b, Pref, t_event=5.4, step=2)
        # Create destination path if it does not already exist
        Path(folder_des).mkdir(parents=True, exist_ok=True)
        # Save features
        np.save(os.path.join(folder_des, filename_des), np.array([dN]))


# ## dFBmelpowfullMAEpos_S

# In[30]:


# Define source folder
sound_dataset = 'projet1_synth_spatmono'
# Create full source folde path
folder_src = os.path.join('data', 'audio_files', sound_dataset)
# Create destination path based on naming convention for the projet
features = 'dFBmelpowfullMAEpos_S'
folder_des = get_features_path(features, sound_dataset, True)


# In[31]:


# Get number of samples in warning signals
y_w, _ = librosa.load(os.path.join(folder_src, file), sr=sr_b)
n_war = y_w.shape[0]


# In[32]:


F_back_list = get_F_back_extracts(cal_back*y_b, sr_b, n_war, step=2, freq_scale="mel")


# In[33]:


for file in tqdm(os.listdir(folder_src)):
    # Create destination file name
    filename_des = os.path.splitext(file)[0] + '.npy'
    # Check if features already computed for this file
    # deja_vu = os.path.exists(os.path.join(folder_des, filename_des))
    # # Compute if not already done
    # if not deja_vu:
    # Load file, upsample to match background sr
    y_w, sr_w = librosa.load(os.path.join(folder_src, file), sr=sr_b)
    # Compute loudness for signal alone
    dF = get_F_diff_wb_full(F_back_list, cal_back*y_w, sr_b, method="MAE", only_pos=True, freq_scale="mel", amp_scale="pow")
    # Create destination path if it does not already exist
    Path(folder_des).mkdir(parents=True, exist_ok=True)
    # Save features
    np.save(os.path.join(folder_des, filename_des), np.array([dF]))


# In[ ]:

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

# roughness computation

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


# if __name__ == "__main__":
#     # Define source folder
sound_dataset = "projet1_synth_spatmono"
# Create full source folder path
folder_src = os.path.join("data", "audio_files", sound_dataset)
file_list = os.listdir(folder_src)
Parallel(n_jobs=4)(delayed(calc_roughness)(file) for file in file_list)





