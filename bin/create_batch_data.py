#!/bin/python 

import os
import sys

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

import tensorflow as tf

from bnn.model import create_encoder_model, encoder_min_input_size
from bnn.util import isAWS, upload_s3, stop_instance, save_pickle_file, BatchConfig, full_path
from bnn.data import test_train_data
from math import ceil

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cifar10', 'The dataset to train the model on.')
flags.DEFINE_string('encoder', 'resnet50', 'The encoder model to train from.')
flags.DEFINE_integer('batch_size', 32, 'The batch size for the generator')
flags.DEFINE_boolean('debug', False, 'If this is for debugging the model/training process or not.')
flags.DEFINE_integer('verbose', 0, 'Whether to use verbose logging when constructing the data object.')
flags.DEFINE_boolean('augment', False, 'Whether to add augmented data to the initial data.')
flags.DEFINE_boolean('stop', True, 'Stop aws instance after finished running.')

def main(_):
	config = BatchConfig(FLAGS.encoder, FLAGS.dataset)
	config.info()

	if os.path.exists(full_path(config.batch_folder())) == False:
		os.makedirs(full_path(config.batch_folder))

	min_image_size = encoder_min_input_size(FLAGS.encoder)
	
	((x_train, y_train), (x_test, y_test)) = test_train_data(FLAGS.dataset, min_image_size, FLAGS.debug, 
		augment_data=FLAGS.augment, batch_size=FLAGS.batch_size)

	input_shape = list(min_image_size)
	input_shape.append(3)

	encoder = create_encoder_model(FLAGS.encoder, input_shape)

	print("Compiling model.")
	encoder.compile(optimizer='sgd', loss='mean_squared_error')

	print("Encoding training data.")
	x_train_encoded = encoder.predict_generator(x_train,
		int(ceil(len(y_train)/FLAGS.batch_size)),
		verbose=FLAGS.verbose)

	print("Encoding test data.")
	x_test_encoded = encoder.predict_generator(x_test,
		int(ceil(len(y_test)/FLAGS.batch_size)),
		verbose=FLAGS.verbose)

	print("Finished encoding data.")

	if FLAGS.augment:
		train_file_name = "/augment-train.p"
		test_file_name = "/augment-test/p"
	else:
		train_file_name = "/train.p"
		test_file_name = "/test/p"

	train_file = config.batch_folder() + train_file_name
	test_file = config.batch_folder() + test_file_name
	save_pickle_file(train_file, (x_train_encoded, y_train))
	save_pickle_file(test_file, (x_test_encoded, y_test))

	if isAWS() and FLAGS.debug == False:
		upload_s3(train_file)
		upload_s3(test_file)

	if isAWS() and FLAGS.stop:
		stop_instance()


if __name__ == '__main__':
	tf.app.run()
