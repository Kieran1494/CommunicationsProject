import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from commpy.filters import rcosfilter, rrcosfilter   # RC / RRC filters
from pulse_shape_detector import detect_pulse_shape, estimate_alpha


def detect_baud_rate_power(received: np.ndarray, fs: float) -> float:
    """
    Rough baud-rate estimate from power spectrum peak.
    """
    N = len(received)

    # Instantaneous power
    power = np.abs(received) ** 4

    # Remove DC
    power -= np.mean(power)

    # FFT and frequency axis
    power_spectrum = np.fft.rfft(power)
    freqs = np.fft.rfftfreq(N, d=1 / fs)  # Hz

    idx_peak = int(np.argmax(np.abs(power_spectrum)))
    baud_est = float(abs(freqs[idx_peak]))
    return baud_est


def detect_baud_rate_autocorr(received: np.ndarray,
                              fs: float,
                              min_baud: float = 100,
                              max_baud: float | None = None) -> float:

    # Envelope / magnitude-only
    x = np.abs(received).astype(float)
    x -= np.mean(x)

    N = len(x)

    # FFT-based autocorrelation
    X = np.fft.fft(x, n=2 * N)
    R = np.fft.ifft(X * np.conj(X)).real
    R = R[:N]

    # Convert baud constraints to SPS constraints
    sps_min = int(fs / (max_baud if max_baud else fs))  # if max_baud None => 1
    sps_max = int(fs / min_baud)

    # At least 2 samples per symbol
    sps_min = max(2, sps_min)

    # Find peaks in autocorrelation within [sps_min, sps_max)
    if sps_min >= sps_max:
        # Fallback: just use global max away from 0 lag
        lag_hat = np.argmax(R[1:]) + 1
        return float(fs / lag_hat)

    peaks, _ = find_peaks(R[sps_min:sps_max])
    peaks = peaks + sps_min

    if len(peaks) < 2:
        # Not enough peaks; fallback
        lag_hat = int(np.argmax(R[1:]) + 1)
        return float(fs / lag_hat)

    # Spacing between peaks ≈ samples-per-symbol
    deltas = np.diff(peaks)
    sps_hat = int(np.median(deltas))

    if sps_hat <= 0:
        # Safety fallback
        sps_hat = int(fs / min_baud)

    baud_hat = fs / sps_hat
    return float(baud_hat)
