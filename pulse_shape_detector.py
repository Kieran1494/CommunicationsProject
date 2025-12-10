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