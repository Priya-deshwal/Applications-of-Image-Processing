import cv2
import numpy as np
import librosa

def calculate_spectral_statistics(spec):
    # Calculate mean, standard deviation, skewness, and kurtosis along the frequency axis
    mean_spectrum = np.mean(spec, axis=1)
    std_spectrum = np.std(spec, axis=1)
    return mean_spectrum, std_spectrum

def solution(audio_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    y, sr = librosa.load(audio_path, sr=None) #sr=None preserves sampling rate
    n_fft = 2048 #FFT points, adjust as you need
    hop_length = 512 #Sliding amount for windowed FFT (adjust as needed)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, fmax=22000)

    mean_spectrum, std_spectrum = calculate_spectral_statistics(spec)
    threshold_mean = 1
    if np.mean(mean_spectrum) > threshold_mean:
        class_name = 'metal'
    else:
        class_name = 'cardboard'
    return class_name
    
