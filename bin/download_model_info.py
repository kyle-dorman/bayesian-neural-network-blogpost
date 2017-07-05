#!/bin/python 

import os
import sys

project_path, x = os.path.split(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)

import tensorflow as tf

from bnn.util import BayesianConfig, BatchConfig, download_s3

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cifar10', 'The dataset to train the model on.')
flags.DEFINE_string('encoder', 'resnet50', 'The encoder model to train from.')
flags.DEFINE_integer('epochs', 1, 'Number of training examples.')
flags.DEFINE_integer('monte_carlo_simulations', 100, 'The number of monte carlo simulations to run for the aleatoric categorical crossentroy loss function.')
flags.DEFINE_integer('batch_size', 32, 'The batch size for the generator')

def main(_):
	bayesian_config = BayesianConfig(FLAGS.encoder, FLAGS.dataset, FLAGS.batch_size, FLAGS.epochs, FLAGS.monte_carlo_simulations)
	bayesian_config.info()

	batch_config = BatchConfig(FLAGS.encoder, FLAGS.dataset)
	batch_config.info()

	print("Downloading model info")

	download_s3(batch_config.batch_folder()+"/train.p")
	download_s3(batch_config.batch_folder()+"/test.p")
	download_s3(batch_config.batch_folder()+"/augment-train.p")
	download_s3(batch_config.batch_folder()+"/augment-test.p")
	download_s3(bayesian_config.model_file())
	download_s3(bayesian_config.csv_log_file())

	print("Done downloading model info")

if __name__ == '__main__':
	tf.app.run()
