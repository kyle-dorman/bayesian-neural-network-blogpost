#!/bin/python 

import os
import sys

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

import tensorflow as tf
from bnn.evaluate import evaluate_model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cifar10', 'The dataset to train the model on.')
flags.DEFINE_string('encoder', 'resnet50', 'The encoder model to train from.')
flags.DEFINE_integer('model_epochs', 1, 'Number of training examples for the saved model.')
flags.DEFINE_integer('monte_carlo_simulations', 100, 'The number of monte carlo simulations to run for the aleatoric categorical crossentroy loss function.')
flags.DEFINE_integer('model_batch_size', 32, 'The batch size for the saved model.')
flags.DEFINE_integer('batch_size', 32, 'The batch size for evaluating model.')
flags.DEFINE_boolean('debug', False, 'If this is for debugging the model/training process or not.')
flags.DEFINE_integer('verbose', 0, 'Whether to use verbose logging when constructing the data object.')
flags.DEFINE_boolean('full_model', False, 'Whether to load the end to end model or just the dense layers.')

def main(_):
	results = evaluate_model(FLAGS.dataset, FLAGS.encoder, FLAGS.model_epochs, 
		FLAGS.monte_carlo_simulations, FLAGS.model_batch_size, FLAGS.batch_size, 
		FLAGS.debug, FLAGS.verbose, FLAGS.full_model)

	print(results)

if __name__ == '__main__':
	tf.app.run()
