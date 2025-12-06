import numpy as np
import commpy

def circ_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.fft.fft(a)
    b = np.fft.fft(np.pad(b, (0, a.size - b.size), mode='constant', constant_values=0))
    return np.fft.ifft(a * b)

def rrc_pulse_shape(data: np.ndarray, oversamples: int, taps: int, alpha:float=0.5):
    _, rrc_taps = commpy.rrcosfilter(taps, alpha, 1, oversamples)
    return circ_convolve(data, rrc_taps)

def estimate_cfo(received: np.ndarray, fs: float, show: bool = False) -> (float, float):
    spectrum = abs(np.fft.fft(np.pad(received**4, (0, received.size), mode='constant'))).astype(np.float128)**2
    mx = np.argmax(spectrum)
    freq = np.fft.fftfreq(spectrum.size, 4 / fs)[mx]
    snr = spectrum[mx] / np.mean(spectrum)
    if show:
        from matplotlib import pyplot as plt
        plt.plot(np.fft.fftfreq(spectrum.size, 4 / fs), spectrum)
        plt.show()
    return freq, snr

def est_eq(received: np.ndarray, pilot: np.ndarray) -> np.complex64:
    h, _, _, _ = np.linalg.lstsq(pilot[:, None], received[:, None], rcond=None)
    return 1.0 / h[0, 0]

def rrc_2d(data: np.ndarray, oversamples: int, taps: int, alpha:float=0.5):
    _, rrc_taps = commpy.rrcosfilter(taps, alpha, 1, oversamples)
    return np.vstack([circ_convolve(data[k, :], rrc_taps) for k in range(data.shape[0])])
