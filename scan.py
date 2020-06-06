import time
import tensorflow as tf
import numpy as np
import os
import sys, argparse
from rtlsdr import RtlSdr
import scipy.signal as signal
from tensorflow import keras
from keras.models import load_model

def read_samples(sdr, freq):
    f_offset = 250000  # shifted tune to avoid DC
    sdr.center_freq = freq - f_offset
    time.sleep(0.06)
    iq_samples = sdr.read_samples(1221376)
    iq_samples = iq_samples[0:600000]
    fc1 = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate * np.arange(len(iq_samples)))  # shift down 250kHz
    iq_samples = iq_samples * fc1
    return iq_samples


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ppm', type=int, default=0,
                    help='dongle ppm error correction')
parser.add_argument('--gain', type=int, default=20,
                    help='dongle gain level')
parser.add_argument('--threshold', type=float, default=0.75,
                    help='threshold to display/hide predictions')
parser.add_argument('--start', type=int, default=85000000,
                    help='begin scan here, in Hertz')
parser.add_argument('--stop', type=int, default=108000000,
                    help='stop scan here, in Hertz')
parser.add_argument('--step', type=int, default=100000,
                    help='step size for scan, in Hertz')


args = parser.parse_args()

model = load_model('fourthModel.h5')
sdr = RtlSdr()
sdr.sample_rate = sample_rate = 2400000
sdr.err_ppm = 56
sdr.gain = 'auto'
#model.save('fourthModel.h5')

train_path = 'training_data'
correct_predictions = 0
classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
num_classes = len(classes)

freq = args.start
while freq <= args.stop:
    samples = []

    iq_samples = read_samples(sdr, freq)
    iq_samples = signal.decimate(iq_samples, 48)

    real = np.real(iq_samples)
    imag = np.imag(iq_samples)

    iq_samples = np.ravel(np.column_stack((real, imag)))
    iq_samples = iq_samples[:1568]

    samples.append(iq_samples)

    samples = np.array(samples)

    # reshape for convolutional model
    samples = np.reshape(samples, (len(samples), 28, 28, 2))
    prediction = model.predict(samples)

    # print predicted label
    maxim = 0.0
    maxlabel = ""
    for sigtype, probability in zip(classes, prediction[0]):
        if probability >= maxim:
            maxim = probability
            maxlabel = sigtype
    print("\n Frequency:    ",  freq / 1e6, "\n classes:      ", classes, " \n probability:  ", prediction[0], '\n PREDICTION:   ' , maxlabel, maxim * 100,"%")
    freq += args.step

sdr.close()


















