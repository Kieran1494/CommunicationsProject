import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
from scipy.signal import find_peaks
import commpy
from commpy.filters import rcosfilter
from commpy.filters import rrcosfilter   # <--- use commpy's RRC

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

def detect_baud_rate_autocorr(received, fs, sps_min=2, sps_max=None):
    x = np.abs(received) ** 2
    x -= np.mean(x)

    N = len(x)

    R_full = np.fft.ifft(
        np.fft.fft(x, n=2*N) * np.conj(np.fft.fft(x, n=2*N))
    )
    R_full = np.real(R_full)
    R = R_full[:N]

    

    if sps_max is None:
        sps_max = N // 20

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


if __name__ == '__main__':
    # -----------------------------
    # Parameters
    # -----------------------------
    Fs = 1_000_000        # sample rate (Hz)
    Rs = 10_860           # baud rate (symbols/s)
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
    t, rrc = rrcosfilter(N_taps, rolloff, 1.0 / Rs, Fs)

    # Pulse shaping (convolution)
    x = np.convolve(upsampled, rrc.astype(np.complex64), mode="same")

    # -----------------------------
    # Test baud-rate detector
    # -----------------------------
    baud_est = detect_baud_rate_autocorr(x, Fs)
    print("True baud:", Rs, "Estimated baud:", baud_est)
