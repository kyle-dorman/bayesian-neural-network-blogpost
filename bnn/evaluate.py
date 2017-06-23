#!/bin/python

from keras.utils.generic_utils import get_custom_objects

from bnn.model import load_baysean_model, load_full_model, encoder_min_input_size
from bnn.data import test_train_batch_data, test_train_data
from bnn.util import BayesianConfig
from bnn.loss_equations import bayesian_categorical_crossentropy

def evaluate_model(dataset, encoder, model_epochs, monte_carlo_simulations, 
	model_batch_size, batch_size, debug, verbose, full_model):
	config = BayesianConfig(encoder, dataset, model_batch_size, model_epochs, monte_carlo_simulations)
	config.info()

	min_image_size = encoder_min_input_size(encoder)

	if full_model:
		((x_train, y_train), (x_test, y_test)) = test_train_data(dataset, min_image_size, debug)
	else:
		((x_train, y_train), (x_test, y_test)) = test_train_batch_data(dataset, encoder, debug)

	min_image_size = list(min_image_size)
	min_image_size.append(3)
	num_classes = y_train.shape[-1]

	model = load_testable_model(encoder, config, monte_carlo_simulations, num_classes, min_image_size, full_model)

	if debug:
		print(model.summary())

	print("Evaluating model training data.")
	train_result = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=verbose)

	print("Evaluating model testing data.")
	test_result = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=verbose)

	print("Finished evaluating model.")
	return (train_result, test_result)

def load_testable_model(encoder, config, monte_carlo_simulations, num_classes, min_image_size, full_model):
	get_custom_objects().update({"bayesian_categorical_crossentropy_internal": bayesian_categorical_crossentropy(monte_carlo_simulations, num_classes)})

	if full_model:
		model = load_full_model(encoder, config.model_file(), min_image_size)
		print("Compiling model.")
		model.compile(
			optimizer='adam',
			loss={'logits_variance': bayesian_categorical_crossentropy(monte_carlo_simulations, num_classes)},
			metrics={'softmax_output': 'categorical_accuracy', 'softmax_output': 'top_k_categorical_accuracy'})
	else:
		model = load_baysean_model(config.model_file())

	return model

def predict(encoder, dataset, debug, monte_carlo_simulations, min_image_size, full_model):
	if full_model:
		((x_train, y_train), (x_test, y_test)) = test_train_data(dataset, min_image_size[0:2], debug)
	else:
		((x_train, y_train), (x_test, y_test)) = test_train_batch_data(dataset, encoder, debug)

	config = BayesianConfig(encoder, dataset, 256, 50, monte_carlo_simulations)
	model = load_testable_model(encoder, config, monte_carlo_simulations, y_train.shape[-1], min_image_size, full_model)

	predictions = model.predict(x_train[0:10])[1]
	labels = y_train[0:10]
	return list(zip(np.argmax(predictions, axis=1), np.argmax(labels, axis=1)))


