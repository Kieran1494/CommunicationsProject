from baud_rate_detector import detect_baud_rate_autocorr
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
        raw, f_delta = gen_digital(keying, order, 1e6, baud, fs, 10, alpha=0.5, center_var=5e3,
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

    error = [0, 0, 0, 0, 0, 0, 0]
    number_of_trials = 100

    for i in range(number_of_trials):
        result = determine_modulation("mask", 4)
        if result != "MASK4":
            error[0] += 1
    print(f"MASK4 Error: {(error[0]/number_of_trials)*100:.2f}%")
    for i in range(number_of_trials):
        result =  determine_modulation("bask", 4)
        if result != "BASK4":
            error[1] += 1
    print(f"BASK4 Error: {(error[1]/number_of_trials)*100:.2f}%")
    for i in range(number_of_trials):
        result = determine_modulation("psk", 8)
        if result != "PSK8":
            error[2] += 1
    print(f"PSK8 Error: {(error[2]/number_of_trials)*100:.2f}%")
    for i in range(number_of_trials):
        result = determine_modulation("psk", 4)
        if result != "PSK4":
            error[3] += 1
    print(f"PSK4 Error: {(error[3]/number_of_trials)*100:.2f}%")
    for i in range(number_of_trials):
        result = determine_modulation("qam", 64)
        if result != "QAM64":
            error[4] += 1
    print(f"QAM64 Error: {(error[4]/number_of_trials)*100:.2f}%")
    for i in range(number_of_trials):
        result = determine_modulation("qam", 32)
        if result != "QAM32":
            error[5] += 1
    print(f"QAM32 Error: {(error[5]/number_of_trials)*100:.2f}%")
    for i in range(number_of_trials):
        result = determine_modulation("qam", 16)
        if result != "QAM16":
            error[6] += 1
    print(f"QAM16 Error: {(error[6]/number_of_trials)*100:.2f}%")

    print(f"Total Error: {(sum(error) / (number_of_trials * 7)) * 100:.2f}%")
