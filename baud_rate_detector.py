import numpy as np
from commpy.filters import rrcosfilter   # <--- use commpy's RRC

def detect_baud_rate(received: np.ndarray, fs: float) -> float:
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
    baud_est = detect_baud_rate(x, Fs)
    print("True baud:", Rs, "Estimated baud:", baud_est)
