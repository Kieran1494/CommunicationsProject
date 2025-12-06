from const_match import estimate_mod
from digital_gen import digital_gen
from monte_carlo import spectra, mix
from pulse_shapper import rrc_pulse_shape, estimate_cfo

from matplotlib import pyplot as plt

if __name__ == '__main__':
    oversamples = 4
    fs = 1e6 # this is fake
    alpha = 0.5

    # data sent
    psk8, f_delta = digital_gen("psk", 8, 1e6, oversamples, 30, alpha=0.5, center_var=0.01)
    spectra(psk8, fs)

    # detect alpha
    detected_alpha = 0.5
    # receive end rrc
    shaped = rrc_pulse_shape(psk8, oversamples, 128, alpha=detected_alpha)

    # detect cfo needs some editing to include oversample timing
    # no need to check aliases
    cfo, snr = estimate_cfo(shaped, 1, True)
    # remove cfo
    offset = 0
    data = mix(shaped, -cfo, 1)[offset::oversamples]

    # now we have a constellation
    plt.scatter(data.real, data.imag)
    plt.show()

    # match constellation
    results = estimate_mod(data)
    print(results)
