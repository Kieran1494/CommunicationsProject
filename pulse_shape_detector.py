import numpy as np
from scipy.signal import welch
from commpy.filters import rcosfilter, rrcosfilter

def estimate_alpha(
    rx: np.ndarray,
    fs: float,
    Rs_est: float,
    nperseg: int = 4096,
    thresh_db: float = -20.0,
) -> float:
    
    # 1) PSD via Welch
    f, S = welch(rx, fs=fs, nperseg=nperseg)
    S = np.abs(S)

    # Normalize PSD to 0 dB peak
    S_db = 10 * np.log10(S / (np.max(S) + 1e-20) + 1e-20)

    # 2) Only look at non-negative frequencies
    mask_pos = f >= 0
    f_pos = f[mask_pos]
    S_db_pos = S_db[mask_pos]

    # 3) Find last frequency where PSD is above threshold (e.g. -20 dB)
    #    That approximates the edge of the occupied band.
    above = S_db_pos > thresh_db
    if not np.any(above):
        # If nothing is above threshold, just bail with alpha=0
        return 0.0

    # Index of the last point still above threshold
    idx_edge = np.where(above)[0][-1]
    f_edge = f_pos[idx_edge]

    # 4) Use B ≈ f_edge ≈ (1 + alpha) * Rs / 2  -->  alpha = 2B/Rs - 1
    alpha_hat = 2.0 * f_edge / Rs_est - 1.0

    # Clip to [0, 1] just in case
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

    f_rx, S_rx = welch(rx, fs=fs, nperseg=nperseg)
    S_rx = np.abs(S_rx)

    sps = fs / Rs_est
    Ts = 1.0 / Rs_est
    N = max(4, int(round(span_symbols * sps)))
    fft_len = 4 * len(f_rx)
    f_H = np.fft.rfftfreq(fft_len, d=1.0 / fs)

    _, h_rc  = rcosfilter(N, alpha, Ts, fs)
    _, h_rrc = rrcosfilter(N, alpha, Ts, fs)

    h_rc  /= np.sqrt(np.sum(np.abs(h_rc)  ** 2))
    h_rrc /= np.sqrt(np.sum(np.abs(h_rrc) ** 2))

    t = (np.arange(N) - N // 2) / fs
    h_sinc = np.sinc(t / Ts)
    h_sinc /= np.sqrt(np.sum(np.abs(h_sinc) ** 2))

    # least squares
    H_rc = np.fft.rfft(h_rc, n=fft_len)
    P_rc = np.abs(H_rc) ** 2
    P_rc_interp = np.interp(f_rx, f_H, P_rc)

    k_rc = np.dot(P_rc_interp, S_rx) / (np.dot(P_rc_interp, P_rc_interp) + 1e-12)
    err_rc = np.sum((S_rx - k_rc * P_rc_interp) ** 2)

    H_rrc = np.fft.rfft(h_rrc, n=fft_len)
    P_rrc = np.abs(H_rrc) ** 2
    P_rrc_interp = np.interp(f_rx, f_H, P_rrc)

    k_rrc = np.dot(P_rrc_interp, S_rx) / (np.dot(P_rrc_interp, P_rrc_interp) + 1e-12)
    err_rrc = np.sum((S_rx - k_rrc * P_rrc_interp) ** 2)

    H_sinc = np.fft.rfft(h_sinc, n=fft_len)
    P_sinc = np.abs(H_sinc) ** 2
    P_sinc_interp = np.interp(f_rx, f_H, P_sinc)

    k_sinc = np.dot(P_sinc_interp, S_rx) / (np.dot(P_sinc_interp, P_sinc_interp) + 1e-12)
    err_sinc = np.sum((S_rx - k_sinc * P_sinc_interp) ** 2)
    #decision
    errs = {
        "RC": err_rc,
        "RRC": err_rrc,
        "SINC": err_sinc,
    }

    best_type = min(errs, key=errs.get)

    return best_type, err_rc, err_rrc, err_sinc
