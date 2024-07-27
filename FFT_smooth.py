import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

def fft_denoise(signal, threshold=0.1):
    signal_fft = fft(signal)
    freqs = np.fft.fftfreq(len(signal))
    signal_fft[np.abs(freqs) > threshold] = 0
    cleaned_signal = ifft(signal_fft)
    return cleaned_signal.real

#data_denoised = data.apply(fft_denoise, threshold=0.05)
