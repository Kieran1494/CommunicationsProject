import matplotlib.pyplot as plt
import numpy as np
import commpy
from commpy.filters import rcosfilter
from commpy.filters import rrcosfilter

from monte_carlo import spectra


def circ_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.fft.fft(a)
    b = np.fft.fft(b, n=a.size)
    return np.fft.ifft(a * b)

def rrc_pulse_shape(data: np.ndarray, oversamples: float, taps: int, alpha:float=0.5):
    _, rrc_taps = rrcosfilter(taps, alpha, 1, oversamples)
    return circ_convolve(data, rrc_taps)

def estimate_cfo(received: np.ndarray, fs: float, oversamples: int, power:int=4, show: bool = False) -> (float, float, int):
    opt_snr = -np.inf
    opt_freq = 0
    opt_offset = 0

    calc_fs = fs / (oversamples * power)

    for offset in range(oversamples):
        spliced_rec = received[offset::oversamples]
        spectrum = abs(np.fft.fft(spliced_rec**power, n=spliced_rec.size*2)).astype(np.float128)**2
        mx = np.argmax(spectrum)
        freq = np.fft.fftfreq(spectrum.size, 1/calc_fs)[mx]
        snr = spectrum[mx] / np.mean(spectrum)
        if snr > opt_snr:
            opt_freq = freq
            opt_snr = snr
            opt_offset = offset
            display_spectrum = spectrum
    if show:
        from matplotlib import pyplot as plt
        plt.plot(np.fft.fftfreq(display_spectrum.size, 1/calc_fs), 10*np.log10(display_spectrum))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.show()
    return opt_freq, opt_snr, opt_offset

def est_eq(received: np.ndarray, pilot: np.ndarray) -> np.complex64:
    h, _, _, _ = np.linalg.lstsq(pilot[:, None], received[:, None], rcond=None)
    return 1.0 / h[0, 0]

def blind_eq(constellation: np.ndarray) -> np.ndarray:
    hist, bins = np.histogram(np.angle(constellation), bins=constellation.size//50)
    mx = np.argmax(hist)
    max_phase = np.mean(bins[mx:mx+1])
    return constellation * np.exp(-1j * max_phase)


def rrc_2d(data: np.ndarray, oversamples: int, taps: int, alpha:float=0.5):
    _, rrc_taps = rrcosfilter(taps, alpha, 1, oversamples)
    return np.vstack([circ_convolve(data[k, :], rrc_taps) for k in range(data.shape[0])])
