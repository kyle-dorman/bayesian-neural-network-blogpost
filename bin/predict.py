#!/bin/python 

import os
import sys

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

import tensorflow as tf
from bnn.predict import predict
from bnn.util import save_pickle_file, full_path

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cifar10', 'The dataset to train the model on.')
flags.DEFINE_string('encoder', 'resnet50', 'The encoder model to train from.')
flags.DEFINE_integer('model_epochs', 1, 'Number of training examples for the saved model.')
flags.DEFINE_integer('train_monte_carlo_simulations', 100, 'The number of monte carlo simulations to run for the aleatoric categorical crossentroy loss function.')
flags.DEFINE_integer('epistemic_monte_carlo_simulations', 100, 'The number of monte carlo simulations to run for the epistemic uncertainty calculation.')
flags.DEFINE_integer('model_batch_size', 32, 'The batch size for the saved model.')
flags.DEFINE_integer('batch_size', 32, 'The batch size for evaluating model.')
flags.DEFINE_integer('verbose', 0, 'Whether to use verbose logging when constructing the data object.')
flags.DEFINE_boolean('debug', False, 'If this is for debugging the model/training process or not.')
flags.DEFINE_boolean('full_model', False, 'Whether to load the end to end model or just the dense layers.')

def main(_):
	(train_results, test_results) = predict(FLAGS.batch_size,
		FLAGS.verbose, FLAGS.epistemic_monte_carlo_simulations,
		FLAGS.debug, FLAGS.full_model,
		FLAGS.encoder, FLAGS.dataset, FLAGS.model_batch_size, 
		FLAGS.model_epochs, FLAGS.train_monte_carlo_simulations)

	print("Done predicting test & train results.")

	if FLAGS.debug == False:
		folder = "predictions/{}_{}".format(FLAGS.encoder, FLAGS.dataset)
		if os.path.isdir(full_path(folder)) == False:
			os.mkdir(full_path(folder))

		save_pickle_file(folder + "/results.p", (train_results, test_results))

if __name__ == '__main__':
	tf.app.run()
