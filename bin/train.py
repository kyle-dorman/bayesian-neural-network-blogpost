#!/bin/python 

import os
import sys

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

import tensorflow as tf
from keras.datasets import cifar10
from keras.optimizers import Adam
from keras.datasets import mnist
import numpy as np
import cv2
from bnn.model import create_model
from bnn.loss_equations import bayesian_categorical_crossentropy

class Config(object):
	def __init__(self, encoder, batch_size, max_epochs):
		self.encoder = encoder
		self.max_epochs = max_epochs
		self.batch_size = batch_size

	def info(self):
		print("encoder:", self.encoder)
		print("batch_size:", self.batch_size)
		print("epochs:", self.max_epochs)

	def model_file(self):
		return "model_{}_{}_{}.ckpt".format(self.encoder, self.batch_size, self.max_epochs)

	def csv_log_file(self):
		return "model_logs_{}_{}_{}.csv".format(self.encoder, self.batch_size, self.max_epochs, self.video_frames, self.min_delta, self.patience)

def one_hot(labels):
	if labels.shape[-1] == 1:
		labels = np.reshape(labels, (-1))
	max_label = np.max(labels) + 1
	return np.eye(max_label)[labels]

def resize(image, shape):
	return cv2.resize(image, shape)

def add_zeros(labels):
	shape = list(labels.shape)
	shape[-1] = 1
	return np.hstack((labels, np.zeros(shape)))

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 32, 'The batch size for the generator')
flags.DEFINE_integer('epochs', 1, 'Number of training examples.')
flags.DEFINE_string('encoder', 'ResNet50', 'The encoder model to train from.')
# flags.DEFINE_float('min_delta', 0.1, 'Early stopping minimum change value.')
# flags.DEFINE_integer('patience', 10, 'Early stopping epochs patience to wait before stopping.')
flags.DEFINE_boolean('verbose', False, 'Whether to use verbose logging when constructing the data object.')
# flags.DEFINE_boolean('stop', True, 'Stop aws instance after finished running.')


def main(_):
	config = Config(FLAGS.encoder, FLAGS.batch_size, FLAGS.epochs)
	config.info()

	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	x_train = np.array([resize(i, (197, 197)) for i in x_train[0:1]])
	y_train = one_hot(y_train)[0:1]

	model = create_model([197,197,3], 10)

	print(model.summary())

	model.compile(optimizer=Adam(lr=1e-4), 
		loss={'logits_variance':bayesian_categorical_crossentropy(100, 10)})

	model.fit(x_train, y_train)


if __name__ == '__main__':
	tf.app.run()
