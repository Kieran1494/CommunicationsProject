import numpy as np
from matplotlib import pyplot as plt

def complex_normal(mean: complex, std: float, size: int | tuple, unit:bool=False) -> np.ndarray:
    if unit and abs(mean) == 0:
        ValueError("Unit specified but mean is 0")
    elif unit:
        mean /= abs(mean)
    return (np.random.normal(mean.real, std, size) + 1j * np.random.normal(mean.imag, std, size)) / np.sqrt(2)

def spectra(data: np.ndarray, fs: float, offset:float=0, show=True, title: str= None):
    ffted = np.fft.fftshift(np.fft.fft(data))
    freqs = np.fft.fftshift(np.fft.fftfreq(data.size, 1/fs)) + offset
    ffted = 10 * np.log10(abs(ffted))
    plt.plot(freqs, ffted)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()

def mix(data: np.ndarray, freq: float, fs: float) -> np.ndarray:
    return data * np.exp(2j * np.pi * freq * np.arange(max(data.shape)) / fs)
