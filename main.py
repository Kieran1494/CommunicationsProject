from baud_rate_detector import detect_baud_rate_autocorr, detect_baud_rate_power
from const_match import estimate_mod, determine_category
from digital_gen import gen_digital
from monte_carlo import spectra, mix
from pulse_shapper import rrc_pulse_shape, estimate_cfo, blind_eq
from pulse_shape_detector import detect_pulse_shape, estimate_alpha
from scipy import signal

from matplotlib import pyplot as plt
import numpy as np
import adi
import time

if __name__ == '__main__':
    # np.random.seed(0)
    fs = 1e6
    baud = fs/6
    alpha = 0.5
    taps = 128 << 1

    for keying, order in (("qam", 16), ):
        # data sent
        print(f"\n{keying}{order}")
        # if "ask" in keying:
            # continue
        raw, f_delta = gen_digital(keying, order, 50e3, baud, fs, 30, alpha=0.5, center_var=1e3,
                                    taps=taps<<3)
        # rx_sdr = adi.Pluto(f"usb:2.5.5")
        # tx_sdr = adi.Pluto(f"usb:2.6.5")
        #
        # # rx_sdr = adi.ad9361(f"ip:{'192.168.40.8'}")
        # # tx_sdr = adi.Pluto(f"ip:{'192.168.40.9'}")
        #
        # # send = np.concatenate([raw, raw, raw])
        # send = raw / max(raw)
        # send *= 2**14
        #
        # rx_sdr.rx_enabled_channels = [0]
        # rx_sdr.sample_rate = int(fs)
        # rx_sdr.rx_rf_bandwidth = int(fs * 0.9)
        # rx_sdr.gain_control_mode_chan0 = 'slow_attack'
        # rx_sdr.gain_control_mode_chan1 = 'slow_attack'
        # rx_sdr.rx_buffer_size = send.size
        # rx_sdr.rx_lo = int(1e9)
        # print(rx_sdr.rx_lo)
        #
        # tx_sdr.tx_enabled_channels = [0]
        # tx_sdr.sample_rate = int(fs)
        # tx_sdr.tx_lo = int(1e9)
        # print(tx_sdr.tx_lo)
        # tx_sdr.tx_hardwaregain_chan0 = -0
        # tx_sdr.tx_cyclic_buffer = True
        # tx_sdr.tx(send)
        # # spectra(raw, fs)
        # print("sleep for 5")
        # time.sleep(5)
        # # rx_sdr.rx()
        # print("done for 5")
        # received = np.asarray(rx_sdr.rx())
        received = raw
        spectra(received, fs)

        #detected_baud = detect_baud_rate_autocorr(received, fs)
        detected_baud = detect_baud_rate_autocorr(received, fs)
        plt.plot(np.fft.rfftfreq(received.size, 1/fs), 20* np.log10(np.abs(np.fft.rfft(abs(received)**2))))
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (dB)")
        plt.title("Frequency vs FFT of Power")
        plt.show()
        print(f"Detected baud rate: {detected_baud} Actual: {baud}")
        ratio = detected_baud / fs


        # detect alpha
        #detected_alpha = estimate_alpha(received, fs, detected_baud)
        detected_alpha = estimate_alpha(received, fs, detected_baud)
        print(f"Detected alpha rate: {detected_alpha} Actual: {alpha}")
        # receive end rrc
        shaped = rrc_pulse_shape(received, ratio, taps, alpha=detected_alpha)
        # spectra(shaped, fs)

        oversamples = 8
        new_fs = oversamples / (fs / detected_baud) * fs
        resampled = signal.resample(shaped, int(received.size * detected_baud / fs * oversamples))
        # spectra(resampled, new_fs)

        # detect cfo needs some editing to include oversample timing
        # no need to check aliases
        cfo, snr, offset = estimate_cfo(resampled, new_fs, oversamples, power=8, show=True)
        print(f"Detected CFO: {cfo} Actual: {f_delta}")
        # remove cfo
        data = mix(resampled, -cfo, new_fs)[offset::oversamples]

        equalized = blind_eq(data)
        print(determine_category(equalized))

        # now we have a constellation
        plt.scatter(equalized.real, equalized.imag, label="Equalized")
        plt.scatter(data.real, data.imag, label="Unequalized")
        plt.legend()
        plt.show()

        # match constellation
        results = estimate_mod(equalized)
        print(results)
