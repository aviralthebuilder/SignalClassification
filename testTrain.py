from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from rtlsdr import RtlSdr
import numpy as np
import scipy.signal as signal
import os 
import time
from tensorflow import keras
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('fifthModel.h5')

sdr = RtlSdr()
sdr.sample_rate = sample_rate = 2400000
sdr.err_ppm = 56
sdr.gain = 'auto'
#model.save('firstModel.h5')

train_path = 'training_data'
correct_predictions = 0
classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
num_classes = len(classes)

def read_samples(freq):
    f_offset = 250000  # shifted tune to avoid DC
    sdr.center_freq = freq - f_offset
    time.sleep(0.06)
    iq_samples = sdr.read_samples(1221376)
    iq_samples = iq_samples[0:600000]
    fc1 = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate * np.arange(len(iq_samples)))  # shift down 250kHz
    iq_samples = iq_samples * fc1
    return iq_samples


def check(freq, corr):
    samples = []
    iq_samples = read_samples(freq)
    iq_samples = signal.decimate(iq_samples, 48, zero_phase=True)

    real = np.real(iq_samples)
    imag = np.imag(iq_samples)

    # iq_samples = np.concatenate((real, imag))
    # iq_samples = np.reshape(iq_samples, (-1, 25000))

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
    print(freq / 1000000, maxlabel, maxim * 100)

    # calculate validation percent
    if corr == maxlabel:
        global correct_predictions
        correct_predictions += 1

plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("ModelAccuracy.png")
# summarize history for loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.savefig("ModelLoss.png")

check(92900000, "wfm")
check(49250000, "tv")
check(95000000, "wfm")
check(104000000, "wfm")
check(482600000, "tv")
check(100500000, "wfm")
check(120000000, "other")
check(106300000, "wfm")
check(942200000, "cellular")
check(107800000, "wfm")

sdr.close()

print("Validation:", correct_predictions / 10 * 100)