#!/bin/python

from bnn.util import open_pickle_file, download_file, unzip_data
from keras.datasets import cifar10
import numpy as np
import cv2

def get_traffic_sign_data():
  url = "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"
  zip_file = "traffic-sign-data.zip"

  download_file(url, zip_file)
  unzip_data(zip_file, "data/traffic-sign")

  train = open_pickle_file("data/traffic-sign/train.p")
  test = open_pickle_file("data/traffic-sign/test.p")
  valid = open_pickle_file("data/traffic-sign/valid.p")

  return ((train['features'], train['labels']), (test['features'], test['labels']), (valid['features'], valid['labels']))

def test_train_data(dataset, min_image_size, isDebug):
	if dataset == 'cifar10':
		(x_train, y_train), (x_test, y_test) = cifar10.load_data()
		return clean_dataset(x_train, y_train, x_test, y_test, min_image_size, isDebug)

	# todo: add more datasets
	else:
		raise ValueError("Unexpected dataset " + dataset + ".")



def clean_dataset(x_train, y_train, x_test, y_test, min_image_size, isDebug):
	if isDebug:
		x_train = x_train[0:128]
		y_train = y_train[0:128]
		x_test = x_test[0:128]
		y_test = y_test[0:128]

	print("Resizing images from", x_train.shape[1:-1], "to", min_image_size)
	x_train = np.array([resize(i, min_image_size) for i in x_train])
	print("Done resizing train images.")
	x_test = np.array([resize(i, min_image_size) for i in x_test])
	print("Done resizing test images.")
	y_train = one_hot(y_train)
	y_test = one_hot(y_test)

	return ((x_train, y_train), (x_test, y_test))

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
