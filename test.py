from baud_rate_detector import detect_baud_rate_autocorr, detect_baud_rate_power
from const_match import estimate_mod, determine_category
from digital_gen import gen_digital
from monte_carlo import spectra, mix
from pulse_shapper import rrc_pulse_shape, estimate_cfo, blind_eq
from scipy import signal

from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    # np.random.seed(0)
    fs = 1e6
    baud = fs/6
    alpha = 0.5
    taps = 128 << 1

    def determine_modulation(keying, order):
        # if "ask" in keying:
            # continue
        raw, f_delta = gen_digital(keying, order, fs/4, baud, fs, 30, alpha=0.5, center_var=5e3,
                                    taps=taps)
        # spectra(raw, fs)

        detected_baud = detect_baud_rate_autocorr(raw, fs)
        ratio = detected_baud / fs


        # detect alpha
        detected_alpha = 0.5
        # receive end rrc
        shaped = rrc_pulse_shape(raw, ratio, taps, alpha=detected_alpha)
        # spectra(shaped, fs)

        oversamples = 8
        new_fs = oversamples / (fs / detected_baud) * fs
        resampled = signal.resample(shaped, int(raw.size * detected_baud / fs * oversamples))
        # spectra(resampled, new_fs)

        # detect cfo needs some editing to include oversample timing
        # no need to check aliases
        cfo, snr, offset = estimate_cfo(resampled, new_fs, oversamples, power=8, show=False)
        # remove cfo
        data = mix(resampled, -cfo, new_fs)[offset::oversamples]

        equalized = blind_eq(data)

        # # now we have a constellation
        # plt.scatter(equalized.real, equalized.imag, color="r")
        # plt.scatter(data.real, data.imag)
        # plt.show()

        # match constellation
        results = estimate_mod(equalized)
        return results

    mod_types = (("mask", 4), ("bask", 4), ("psk", 8), ("psk", 4), ("mask", 2), ("bask", 2), ("qam", 64), ("qam", 32), ("qam", 16))
    error = {f"{keying}{order}": 0 for keying, order in mod_types}
    number_of_trials = 30

    for keying, order in mod_types:
        for i in range(number_of_trials):
            result = determine_modulation(keying, order)
            if result != f"{keying}{order}".upper():
                error[f"{keying}{order}"] += 1
        print(f"{keying}{order} Error: {(error[f"{keying}{order}"] / number_of_trials):%}")

    print(f"Total Error: {(sum(error.values()) / (number_of_trials * len(error.values()))):%}")
