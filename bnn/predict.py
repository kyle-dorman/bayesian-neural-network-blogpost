#!/bin/python

from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
import numpy as np
import math

from bnn.model import load_baysean_model, load_full_model, encoder_min_input_size, load_epistemic_uncertainty_model
from bnn.data import test_train_batch_data, test_train_data
from bnn.util import BayesianConfig
from bnn.loss_equations import bayesian_categorical_crossentropy


def load_testable_model(encoder, config, monte_carlo_simulations, num_classes, min_image_size, full_model):
	if full_model:
		model = load_full_model(encoder, config.model_file(), min_image_size)
		print("Compiling full testable model.")
		model.compile(
			optimizer='adam',
			loss={'logits_variance': bayesian_categorical_crossentropy(monte_carlo_simulations, num_classes)},
			metrics={'softmax_output': ['categorical_accuracy', 'top_k_categorical_accuracy']})
	else:
		model = load_baysean_model(config.model_file())

	return model   


def load_testable_epistemic_uncertainty_model(full_model, min_image_size, config, epistemic_monte_carlo_simulations):
	if full_model:
		model = load_full_epistemic_uncertainty_model(config.encoder, (), config.model_file(), epistemic_monte_carlo_simulations)
	else:
		model = load_epistemic_uncertainty_model(config.model_file(), epistemic_monte_carlo_simulations)

	# the model won't be used for training
	model.compile('adam', 'categorical_crossentropy')
	return model


def predict_epistemic_uncertainties(batch_size, verbose, epistemic_monte_carlo_simulations, debug, full_model,
	x_train, y_train, x_test, y_test,
	encoder, dataset, model_batch_size, model_epochs, model_monte_carlo_simulations):
	# set learning phase to 1 so that Dropout is on. In keras master you can set this
	# on the TimeDistributed layer
	K.set_learning_phase(1)
	min_image_size = encoder_min_input_size(encoder)

	config = BayesianConfig(encoder, dataset, model_batch_size, model_epochs, model_monte_carlo_simulations)
	epistemic_model = load_testable_epistemic_uncertainty_model(full_model, min_image_size, config, epistemic_monte_carlo_simulations)

	# Shape (N)
	print("Predicting epistemic_uncertainties.")
	if hasattr(x_train, 'shape'):
		epistemic_uncertainties_train = epistemic_model.predict(x_train, batch_size=batch_size, verbose=verbose)[0]
		epistemic_uncertainties_test = epistemic_model.predict(x_test, batch_size=batch_size, verbose=verbose)[0]
	else:
		# generator
		epistemic_uncertainties_train = epistemic_model.predict_generator(x_train, int(math.ceil(len(y_train/batch_size))), verbose=verbose)[0]
		epistemic_uncertainties_test = epistemic_model.predict_generator(x_test, int(math.ceil(len(y_test/batch_size))), verbose=verbose)[0]
	
	return (epistemic_uncertainties_train, epistemic_uncertainties_test)


def predict_softmax_aleatoric_uncertainties(batch_size, verbose, debug, full_model,
	x_train, y_train, x_test, y_test,
	encoder, dataset, model_batch_size, model_epochs, model_monte_carlo_simulations):

	num_classes = len(y_train[0])
	min_image_size = encoder_min_input_size(encoder)
	min_image_size = list(min_image_size)
	min_image_size.append(3)
	config = BayesianConfig(encoder, dataset, model_batch_size, model_epochs, model_monte_carlo_simulations)
	model = load_testable_model(encoder, config, model_monte_carlo_simulations, num_classes, min_image_size, full_model)

	print("Predicting softmax and aleatoric_uncertainties.")
	if hasattr(x_train, 'shape'):
		predictions_train = model.predict(x_train, batch_size=batch_size, verbose=verbose)
		predictions_test = model.predict(x_test, batch_size=batch_size, verbose=verbose)	
	else:
		# generator
		predictions_train = model.predict_generator(x_train, int(math.ceil(len(y_train/batch_size))), verbose=verbose)
		predictions_test = model.predict_generator(x_test, int(math.ceil(len(y_test/batch_size))), verbose=verbose)	

	# Shape (N)
	aleatoric_uncertainties_train = np.reshape(predictions_train[0][:,num_classes:], (-1))
	aleatoric_uncertainties_test = np.reshape(predictions_test[0][:,num_classes:], (-1))

	logits_train = predictions_train[0][:,0:num_classes]
	logits_test = predictions_test[0][:,0:num_classes]
	
	# Shape (N, C)
	softmax_train = predictions_train[1]
	softmax_test = predictions_test[1]

	p_train = np.argmax(softmax_train, axis=1)
	p_test = np.argmax(softmax_test, axis=1)
	l_train = np.argmax(y_train, axis=1)
	l_test = np.argmax(y_test, axis=1)
	# Shape (N)
	prediction_comparision_train = np.equal(p_train,l_train).astype(int)
	prediction_comparision_test = np.equal(p_test,l_test).astype(int)

	train_results = [{
		'softmax_raw':softmax_train[i],
		'softmax':p_train[i],
		'logits_raw': logits_train[i],
		'label': np.argmax(y_train[i]),
		'label_expanded':y_train[i],
		'aleatoric_uncertainty':aleatoric_uncertainties_train[i],
		'is_correct':prediction_comparision_train[i]
		} for i in range(len(prediction_comparision_train))]

	test_results = [{
		'softmax_raw':softmax_test[i],
		'softmax':p_test[i],
		'logits_raw': logits_test[i],
		'label': np.argmax(y_test[i]),
		'label_expanded':y_test[i],
		'aleatoric_uncertainty':aleatoric_uncertainties_test[i],
		'is_correct':prediction_comparision_test[i]
		} for i in range(len(prediction_comparision_test))]

	return (train_results, test_results)

def predict_on_data(batch_size, verbose, epistemic_monte_carlo_simulations, debug, full_model,
	x_train, y_train, x_test, y_test,
	encoder, dataset, model_batch_size, model_epochs, model_monte_carlo_simulations, include_epistemic_uncertainty=True):

	(train_results, test_results) = predict_softmax_aleatoric_uncertainties(batch_size, verbose, debug, full_model, 
		x_train, y_train, x_test, y_test,
		encoder, dataset, model_batch_size, model_epochs, model_monte_carlo_simulations)

	# epistemic_uncertainty takes a long time to predict
	if include_epistemic_uncertainty:
		(epistemic_uncertainties_train, epistemic_uncertainties_test) = predict_epistemic_uncertainties(
			batch_size, verbose, epistemic_monte_carlo_simulations, debug, full_model, 
			x_train, y_train, x_test, y_test,
			encoder, dataset, model_batch_size, model_epochs, model_monte_carlo_simulations)

		for i in range(len(epistemic_uncertainties_train)):
			train_results[i]['epistemic_uncertainty'] = epistemic_uncertainties_train[i]

		for i in range(len(epistemic_uncertainties_test)):
			test_results[i]['epistemic_uncertainty'] = epistemic_uncertainties_test[i]

	return (train_results, test_results)


def predict(batch_size, verbose, epistemic_monte_carlo_simulations, debug, full_model,
	encoder, dataset, model_batch_size, model_epochs, model_monte_carlo_simulations):

	min_image_size = encoder_min_input_size(encoder)
	if full_model:
		((x_train, y_train), (x_test, y_test)) = test_train_data(dataset, min_image_size[0:2], 
			debug, augment_data=False, batch_size=batch_size)
	else:
		((x_train, y_train), (x_test, y_test)) = test_train_batch_data(dataset, encoder, debug)

	return predict_on_data(batch_size, verbose, epistemic_monte_carlo_simulations, debug, full_model,
		x_train, y_train, x_test, y_test,
		encoder, dataset, model_batch_size, model_epochs, model_monte_carlo_simulations)


