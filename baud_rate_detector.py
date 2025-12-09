import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from scipy.signal import find_peaks
import commpy
from commpy.filters import rcosfilter
from commpy.filters import rrcosfilter   # <--- use commpy's RRC
from pulse_shape_detector import detect_pulse_shape
from pulse_shape_detector import estimate_alpha

def detect_baud_rate_power(received: np.ndarray, fs: float) -> float:
    N = len(received)

    # Power of received signal
    power = np.abs(received) ** 2

    # Remove DC
    power -= np.mean(power)

    # FFT and frequency axis
    power_spectrum = np.fft.rfft(power)
    freqs = np.fft.rfftfreq(N, d=1/fs)  # Hz

    idx_peak = np.abs(np.argmax(np.abs(power_spectrum)))
    baud_est = np.abs(freqs[idx_peak])
    return baud_est

def detect_baud_rate_autocorr(x, fs, min_baud=100, max_baud=None):
    x = np.abs(x)
    x -= np.mean(x)

    N = len(x)
    X = np.fft.fft(x, n=2*N)
    R = np.fft.ifft(X * np.conj(X)).real
    R = R[:N]

    sps_min = int(fs / (max_baud if max_baud else fs))
    sps_max = int(fs / min_baud)

    sps_min = max(2, sps_min)

    peaks, _ = find_peaks(R[sps_min:sps_max])
    peaks += sps_min

<<<<<<< HEAD
    deltas = np.diff(peaks)
    sps_hat = int(np.median(deltas))
    baud_hat = fs / sps_hat
=======
    lag_min = max(sps_min, 1)
    lag_max = min(sps_max, N - 1)

    search_region = R[lag_min:lag_max]
    peaks, _ = find_peaks(search_region)
    mid_peak = int(math.ceil(len(peaks)/2))
    N_s = peaks[mid_peak-1]-peaks[mid_peak-2]
    # print(N_s)
    baud_est = fs/N_s

    '''
    plt.figure()
    plt.plot(search_region[:700])
    plt.axvline(Fs/Rs, color='r', linestyle='--', label='True sps')
    plt.title("Autocorrelation R[k] (first 500 lags)")
    plt.xlabel("Lag (samples)")
    plt.ylabel("R[k]")
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
    
    return float(baud_est)
>>>>>>> 46e599cb46a42c14823756b4560c625c47a3c2a9

    return float(baud_hat)

if __name__ == '__main__':
    # -----------------------------
    # Parameters
    # -----------------------------
    Fs = 1_000_000        # sample rate (Hz)
    Rs = 10_000           # baud rate (symbols/s)
    sps = int(Fs / Rs)    # samples per symbol (must be integer)
    Nsym = 4000           # number of symbols

    rolloff = 0.35        # RRC roll-off
    span = 8              # filter span in symbols

    # Random QPSK symbols
    symbols = np.random.choice(
        [1+1j, 1-1j, -1+1j, -1-1j],
        size=Nsym
    )

    # Upsample by inserting zeros
    upsampled = np.zeros(Nsym * sps, dtype=np.complex64)
    upsampled[::sps] = symbols

    # RRC filter using commpy
    N_taps = span * sps + 1
    t, rrc = rcosfilter(N_taps, rolloff, 1.0 / Rs, Fs)

    # Pulse shaping (convolution)
    x = np.convolve(upsampled, rrc.astype(np.complex64), mode="same")
    # -----------------------------
    # Test baud-rate detector
    # -----------------------------
    baud_est = detect_baud_rate_autocorr(x, Fs)
    alpha = estimate_alpha(x, Fs, baud_est)
    pulse_est = detect_pulse_shape(x, Fs, baud_est, alpha)
    print(pulse_est)
    print(alpha)
    print("True baud:", Rs, "Estimated baud:", baud_est)
