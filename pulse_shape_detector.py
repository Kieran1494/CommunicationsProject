import numpy as np
from scipy.signal import welch
from commpy.filters import rcosfilter, rrcosfilter

import numpy as np
from scipy.signal import welch

def estimate_alpha(
    rx: np.ndarray,
    fs: float,
    Rs: float,
    nperseg: int = 4096,
    thresh_db: float = -25.0,
) -> float:

    # 1) PSD via Welch
    f, S = welch(rx, fs=fs, nperseg=nperseg)
    S = np.abs(S)
    S /= np.max(S) + 1e-20
    S_db = 10 * np.log10(S + 1e-20)

    # 2) Non-negative freqs only
    mask = f >= 0
    f_pos = f[mask]
    S_db_pos = S_db[mask]

    # 3) Find band-edge where PSD drops below thresh_db and stays low
    above = S_db_pos > thresh_db
    if not np.any(above):
        return 0.0  # no information

    idx_edge = np.where(above)[0][-1]
    f_edge = f_pos[idx_edge]

    # 4) alpha ≈ 2B/Rs - 1
    alpha_hat = 2.0 * f_edge / Rs - 1.0
    alpha_hat = float(np.clip(alpha_hat, 0.0, 1.0))
    return alpha_hat

def detect_pulse_shape(
    rx: np.ndarray,
    fs: float,
    Rs_est: float,
    alpha: float,
    span_symbols: int = 8,
    nperseg: int = 4096,
):


    # PSD of rx
    f_rx, S_rx = welch(rx, fs=fs, nperseg=nperseg)
    S_rx = np.abs(S_rx)

    sps = fs / Rs_est
    Ts = 1.0 / Rs_est
    N = max(4, int(round(span_symbols * sps)))
    fft_len = 4 * len(f_rx)
    f_H = np.fft.rfftfreq(fft_len, d=1.0 / fs)

    # --- RC ---
    _, h_rc = rcosfilter(N, alpha, Ts, fs)
    h_rc /= np.sqrt(np.sum(np.abs(h_rc) ** 2))

    H_rc = np.fft.rfft(h_rc, n=fft_len)
    P_rc = np.abs(H_rc) ** 2
    P_rc_interp = np.interp(f_rx, f_H, P_rc)

    k_rc = np.dot(P_rc_interp, S_rx) / (np.dot(P_rc_interp, P_rc_interp) + 1e-12)
    err_rc = np.sum((S_rx - k_rc * P_rc_interp) ** 2)

    # --- RRC ---
    _, h_rrc = rrcosfilter(N, alpha, Ts, fs)
    h_rrc /= np.sqrt(np.sum(np.abs(h_rrc) ** 2))

    H_rrc = np.fft.rfft(h_rrc, n=fft_len)
    P_rrc = np.abs(H_rrc) ** 2
    P_rrc_interp = np.interp(f_rx, f_H, P_rrc)

    k_rrc = np.dot(P_rrc_interp, S_rx) / (np.dot(P_rrc_interp, P_rrc_interp) + 1e-12)
    err_rrc = np.sum((S_rx - k_rrc * P_rrc_interp) ** 2)

    best_type = "RC" if err_rc < err_rrc else "RRC"
    return best_type, err_rc, err_rrc