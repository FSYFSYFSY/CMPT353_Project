import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import sys

def fft_denoise(signal, threshold=0.1):
    signal_fft = fft(signal, axis=0)
    freqs = np.fft.fftfreq(len(signal))
    signal_fft[np.abs(freqs) > threshold] = 0
    cleaned_signal = ifft(signal_fft)
    return cleaned_signal.real

file_name = sys.argv[1]
df = pd.read_csv(f"Data/{file_name}")

print(df)

data_denoised = fft_denoise(df['ax'].values, threshold=0.2)

df['ax_denoised'] = data_denoised

print(df)

