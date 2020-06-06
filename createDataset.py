import glob
import sklearn
import numpy as np
import scipy.signal as signal
import os

def read_train_data(path, classes, val_size):
	class DataSets(object):
		pass

	dataset = DataSets()

	img, labels, img_names, cls = load_train(path, classes)
	img, labels, img_names, cls = sklearn.utils.shuffle(img, labels, img_names, cls)

	if isinstance(val_size, float): 
		val_size = int(val_size * img.shape[0])

	val_img = img[:val_size]
	val_labels = labels[:val_size]
	val_img_names = img_names[:val_size]
	val_cls = cls[:val_size]

	train_img = img[val_size:]
	train_labels = labels[val_size:]
	train_img_names = img_names[val_size:]
	train_cls = cls[val_size:]

	dataset.train = DataSet2(train_img, train_labels, train_img_names, train_cls)
	dataset.val = DataSet2(val_img, val_labels, val_img_names, val_cls)

	return dataset

def load_train(training_path, classes):
	# for keeping track of the class
	labels = []
	# keeping track of the file name
	sample_names = []
	# keeping track of the file content
	samples = []
	cls = []

	for c in classes:
		index = classes.index(c)
		path = os.path.join(training_path, c, '*.npy')
		files = glob.glob(path)
		for f in files:
			iq_samples = np.load(f)
			# taking the real part
			real = np.real(iq_samples)
			# taking the imaginary path
			imag = np.imag(iq_samples)
			# combines the two arrays to make a 2d array and then flatten it
			iq_samples = np.ravel(np.column_stack((real, imag)))


			multiple = True
			if multiple:
				iq_samples1 = iq_samples[:1568]
				iq_samples1 = iq_samples1.reshape(28, 28, 2)
				iq_samples2 = iq_samples[1568:3136]
				iq_samples2 = iq_samples2.reshape(28, 28, 2)
				iq_samples3 = iq_samples[3136:4704]
				iq_samples3 = iq_samples3.reshape(28, 28, 2)
				iq_samples4 = iq_samples[4704:6272]
				iq_samples4 = iq_samples4.reshape(28, 28, 2)
				samples.append(iq_samples1)
				samples.append(iq_samples2)
				samples.append(iq_samples3)
				samples.append(iq_samples4)

				file_name = os.path.basename(f)
				label = np.zeros(len(classes))
				label[index] = 1.0
				labels.append(label)
				labels.append(label)
				labels.append(label)
				labels.append(label)
				sample_names.append(file_name)
				sample_names.append(file_name)
				sample_names.append(file_name)
				sample_names.append(file_name)
				cls.append(c)
				cls.append(c)
				cls.append(c)
				cls.append(c)
			else:
				iq_samples = iq_samples[:1568]
				iq_samples = np.reshape(iq_samples, (28, 28, 2))
				samples.append(iq_samples)
				label = np.zeros(len(classes))
				label[index] = 1.0
				labels.append(label)
				file_name = os.path.basename(f)
				sample_names.append(file_name)
				cls.append(c)

			# iq_samples = iq_samples[:24576]
			# iq_samples = np.reshape(iq_samples, (-1, 128, 2))
			# samples.append(iq_samples)
			# label = np.zeros(len(classes))
			# label[index] = 1.0
			# labels.append(label)
			# file_name = os.path.basename(f)
			# sample_names.append(file_name)
			# cls.append(c)

	samples = np.array(samples)
	labels = np.array(labels)
	sample_names = np.array(sample_names)
	cls = np.array(cls)
	return samples, labels, sample_names, cls

class DataSet2(object):
	def __init__(self, img, labels, img_names, cls):
		self.number = img.shape[0]
		self.img = img
		self.labels = labels
		self.img_names = img_names
		self.cls = cls
		self.epoch_done = 0
		self.index_epoch = 0

	def images(self):
		return self.img

	def labels(self):
		return self.labels

	def img_names(self):
		return self.img_names

	def cls(self):
		return self.cls

	def number(self):
		return self.number

	def epoch_done(self):
		return self.epoch_done

	def next_batcg(self, batch):
		src = self.index_epoch
		self.index_epoch += batch

		if(self.index_epoch > self.number):
			self.epoch_done += 1
			src = 0
			self.index_epoch = batch
			assert batch <= self.number

		dst = self.index_epoch

		return self.img[src:dts], self.label[src:dst], self.img_names[src:dst], self.cls[src:dst]






































