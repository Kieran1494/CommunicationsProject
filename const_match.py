import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

def determine_category(constellation: np.ndarray):
    mag = abs(constellation)
    pks = peak_detect(mag / max(mag), 0.66, distance=0.2)
    if len(pks) > 1:
        return "ask"
    else:
        pks = peak_detect(mag / max(mag), 0.05, distance=0.2, bin_dec=500)
        if len(pks) > 1:
            return "qam"
        else:
            return "psk"


def peak_detect(data, prominence=0.1, bin_dec=1000, distance=None, visualize=False):
    counts, bin_edges = np.histogram(data, bins=data.size//bin_dec, density=True)
    counts_norm = counts / np.max(counts)
    if distance is not None:
        distance = distance * counts_norm.size
    peaks, properties = signal.find_peaks(counts_norm, prominence=prominence, distance=distance)

    # # Optional: Visualization
    if visualize:
        plt.figure(figsize=(10, 5))
        # Plot histogram
        plt.bar(bin_edges[:-1], counts_norm, width=np.diff(bin_edges), align='edge', alpha=0.6, label='Histogram')
        # Plot peaks
        plt.plot(bin_edges[peaks], counts_norm[peaks], "x", color='red', markersize=10, label='Detected Peaks')
        plt.title(f"Peak Detection: {len(peaks)} Peak(s) Found")
        plt.legend()
        plt.show()

    return peaks / counts_norm.size

def estimate_mod(constellation: np.ndarray)-> dict:
    return {
        "psk8": 1,
        "ask2": 0,
    }