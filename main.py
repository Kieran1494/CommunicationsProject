from baud_rate_detector import detect_baud_rate
from const_match import estimate_mod
from digital_gen import digital_gen, gen_digital
from monte_carlo import spectra, mix
from pulse_shapper import rrc_pulse_shape, estimate_cfo
from scipy import signal

from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    np.random.seed(0)
    fs = 1e6
    baud = fs/6
    alpha = 0.5
    taps = 128 << 1

    # data sent
    psk8, f_delta = gen_digital("psk", 8, 1e6, baud, fs, 30, alpha=0.5, center_var=5e3,
                                taps=taps)
    spectra(psk8, fs)

    detected_baud = detect_baud_rate(psk8, fs)
    print(f"Detected baud rate: {detected_baud} Actual: {baud}")
    ratio = detected_baud / fs


    # detect alpha
    detected_alpha = 0.5
    print(f"Detected alpha rate: {detected_alpha} Actual: {alpha}")
    # receive end rrc
    shaped = rrc_pulse_shape(psk8, ratio, taps, alpha=detected_alpha)
    spectra(shaped, fs)

    oversamples = 4
    new_fs = oversamples / (fs / detected_baud) * fs
    resampled = signal.resample(shaped, int(psk8.size * detected_baud / fs * oversamples))
    spectra(resampled, new_fs)

    # detect cfo needs some editing to include oversample timing
    # no need to check aliases
    cfo, snr, offset = estimate_cfo(resampled, new_fs, oversamples, power=8, show=True)
    print(f"Detected CFO: {cfo} Actual: {f_delta}")
    # remove cfo
    data = mix(resampled, -cfo, new_fs)[offset::oversamples]

    # now we have a constellation
    plt.scatter(data.real, data.imag)
    plt.show()

    # match constellation
    results = estimate_mod(data)
    print(results)
