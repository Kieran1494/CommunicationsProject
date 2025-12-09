import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

def optimal_rotation(constellation: np.ndarray, increments=50, type="nonask") -> np.ndarray:
    angles = np.linspace(-np.pi / 4, np.pi / 4, increments)
    rotated_opt = constellation
    if type == "nonask":
        span_opt = float('inf')
        for theta in angles:
            rotated = constellation * np.exp(-1j * theta)
            span = np.ptp(rotated.real) + np.ptp(rotated.imag)
            if span < span_opt:
                span_opt = span
                rotated_opt = rotated
    else:
        imag_span_opt = float('inf')
        for theta in angles:
            rotated = constellation * np.exp(-1j * theta)
            imag_span = np.percentile(rotated.imag, 95) - np.percentile(rotated.imag, 5)
            if imag_span < imag_span_opt:
                imag_span_opt = imag_span
                rotated_opt = rotated
    return rotated_opt


def histogram_fraction(data, bin_dec=1000, threshold=0.1):
    counts, bin_edges = np.histogram(data, bins=data.size // bin_dec, density=True)
    counts_norm = counts / np.max(counts)
    fraction = np.sum(counts_norm > threshold) / counts.size
    return fraction

def determine_category(constellation: np.ndarray, visualize =False):
    mag = abs(constellation)
    real_span = np.percentile(constellation.real, 95) - np.percentile(constellation.real, 5)
    imag_span = np.percentile(constellation.imag, 95) - np.percentile(constellation.imag, 5)
    ratio = real_span / (imag_span + 1e-12)
    if ratio > 4:
        return "ask"
    else:
        histo_span = histogram_fraction(mag, bin_dec=1000, threshold=0.1)
        if histo_span > 0.5:
            return "qam"
        pks = peak_detect(mag / max(mag), 0.05, distance=0.2, bin_dec=500, visualize=visualize)
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

def estimate_mod(constellation: np.ndarray)-> str:
    type = determine_category(constellation)
    if type == "ask":
        rotated = optimal_rotation(constellation, type="ask")
        real_values = constellation.real
        if np.mean(real_values) > np.std(real_values) / 2:
            tp = "MASK"
        else:
            tp = "BASK"
        angle_peaks = peak_detect(real_values, distance=0.1)
        rough_order = len(angle_peaks)
        error = 20
        for i in [2, 4, 8]:
            current_error = abs(rough_order - i)
            if current_error < error:
                error = current_error
                order = i
        return tp + str(order)

        # threshold = np.percentile(real_values, 20)
        # low_vals = real_values[real_values <= threshold]
        # if np.all(low_vals < 0):
        #     return "BASK4"
        # else:
        #     return "MASK4"
    if type == "qam":
        rotated = optimal_rotation(constellation)
        horizontal_peaks = len(peak_detect(rotated.real, prominence=0.1, bin_dec=500, visualize=False))
        vertical_peaks = len(peak_detect(rotated.imag, prominence=0.1, bin_dec=500, visualize=False))
        rough_order = horizontal_peaks * vertical_peaks
        error = float('inf')
        for i in [16, 32, 64]:
            current_error = abs(rough_order - i)
            if current_error < error:
                error = current_error
                order = i
        return "QAM" + str(order)
    if type == "psk":
        angles = np.angle(constellation * (1+1j))
        angle_peaks = peak_detect(angles)
        rough_order = len(angle_peaks)
        error = 20
        for i in [4, 8]:
            current_error = abs(rough_order - i)
            if current_error < error:
                error = current_error
                order = i
        return "PSK" + str(order)
    else:
        return "Inconclusive"
