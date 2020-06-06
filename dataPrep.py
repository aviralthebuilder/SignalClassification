from rtlsdr import RtlSdr
import time, random, string, os
import numpy as np
import scipy.signal as signal


#anti-aliasing filter is a filter used fore a signal sampler to restrict the bandwidth of a signal
step = 100000
sdr = RtlSdr()
sdr.sample_rate = sample_rate = 2400000
decimation_rate = 48
sdr.err_ppm = 56
sdr.gain = 'auto'

def read_samples(sdr, freq):
    f_offset = 250000  
    sdr.center_freq = freq - f_offset
    time.sleep(0.06)
    iq_samples = sdr.read_samples(1221376)
    iq_samples = iq_samples[0:600000]
    fc1 = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate * np.arange(len(iq_samples)))  
    iq_samples = iq_samples * fc1
    return iq_samples

def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def collect_samples(freq, classname):
    os.makedirs("training_data/" + classname, exist_ok=True)
    os.makedirs("testing_data/" + classname, exist_ok=True)
    for i in range(0, 1000):
        iq_samples = read_samples(sdr, freq)
        freq = freq + step
        iq_samples = signal.decimate(iq_samples, decimation_rate, zero_phase=True)
        if (i < 750):  # 75% train, 25% test
            filename = "training_data/" + classname + "/samples-" + randomword(16) + ".npy"
        else:
            filename = "testing_data/" + classname + "/samples-" + randomword(16) + ".npy"
        np.save(filename, iq_samples)
        if not (i % 10): print(i / 10, "%", classname)

# collect_samples(422600000, "tetra")
collect_samples(95000000, "wfm")
#collect_samples(104000000, "wfm")
collect_samples(942200000, "cellular")
#collect_samples(900000000, "cellular")
collect_samples(147337500, "dmr")
collect_samples(49250000, "tv")
#collect_samples(82250000, "tv")
collect_samples(174000000, "other")