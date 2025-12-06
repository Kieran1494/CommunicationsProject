import numpy as np

from pulse_shapper import rrc_pulse_shape
from monte_carlo import mix, complex_normal


def gen_const(keying:str, order: int, length:int) -> np.ndarray:
    keying = keying.upper()
    targets = []
    if keying == 'QAM':
        targets = []
        table = int(np.ceil(order ** (1 / 2) - 1)) + 1
        sq = table ** 2
        val = np.sqrt((sq - order) / 4)
        while val != int(val):
            table += 1
            sq = table ** 2
            val = np.sqrt((sq - order) / 4)
            if table > 1e3:
                raise Exception(f"Invalid Order: {order}")
        for i in range(-table + 1, table, 2):
            for j in range(-table + 1, table, 2):
                # if  val:
                if abs(i) > table - 2 * val and abs(j) > table - 2 * val:
                    continue
                targets.append(i + 1j * j)
    elif keying == 'PSK':
        targets = np.exp(1j * np.linspace(0, 2 * np.pi, order, endpoint=False))
    elif keying == 'MASK': # monopolar ask
        targets = np.linspace(0, 1, order)
    elif keying == 'BASK': # bipolar ask
        targets = np.linspace(-1, 1, order)

    targets = np.array(targets)
    targets /= np.mean(np.abs(targets) ** 2)

    return np.random.choice(targets, size=int(length))

def digital_gen(keying:str, order: int, length:int, repeat:int, snr:float, walk:float=0, center_var:float = 0,
                dtype=np.complex64, taps=128, alpha=0.5) -> (np.ndarray, float):
    data = gen_const(keying, order, length // repeat + 1).repeat(repeat)
    data = rrc_pulse_shape(data, repeat, taps, alpha)
    f_shift = np.random.uniform(-center_var, center_var)
    data = mix(data, f_shift, 1)

    channel = complex_normal(0, 1, 1)[0]
    channel += np.cumsum(complex_normal(0, walk, data.size)) # don't use this unless you know what you are doing
    data *= channel

    sigma = 10 ** (-snr / 10)
    data += complex_normal(0, sigma, data.size)
    return data.astype(dtype)[:int(length)], f_shift
