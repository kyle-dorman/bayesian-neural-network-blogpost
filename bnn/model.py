#!/bin/python 

from keras.applications.resnet50 import ResNet50
import numpy as np
# from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, Dropout, Activation, Lambda, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.engine.topology import Layer
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from bnn.loss_equations import bayesian_categorical_crossentropy

# Take a mean of the results of a TimeDistributed layer.
# Applying TimeDistributedMean()(TimeDistributed(T)(x)) to an
# input of shape (None, ...) returns putpur of same size.
class TimeDistributedMean(Layer):
	def build(self, input_shape):
		super(TimeDistributedMean, self).build(input_shape)

	# input shape (None, T, ...)
	# output shape (None, ...)
	def compute_output_shape(self, input_shape):
		return (input_shape[0],) + input_shape[2:]

	def call(self, x):
		return K.mean(x, axis=1)


# Apply the predictive entropy function for input with C classes. 
# Input of shape (None, C, ...) returns output with shape (None, ...)
# Input should be predictive means for the C classes.
# In the case of a single classification, output will be (None,).
class PredictiveEntropy(Layer):
	def build(self, input_shape):
		super(PredictiveEntropy, self).build(input_shape)

	# input shape (None, C, ...)
	# output shape (None, ...)
	def compute_output_shape(self, input_shape):
		return (input_shape[0],)

	# x - prediction probability for each class(C)
	def call(self, x):
		return -1 * K.sum(K.log(x) * x, axis=1)


def load_full_model(encoder, checkpoint, input_shape):
	encoder_model = create_encoder_model(encoder, input_shape)
	bayesian_model = load_bayesian_model(checkpoint)
	outputs = bayesian_model(encoder_model.outputs)
	# hack to rename outputs
	logits_variance = Lambda(lambda x: x, name='logits_variance')(outputs[0])
	softmax_output = Lambda(lambda x: x, name='softmax_output')(outputs[1])

	return Model(inputs=encoder_model.inputs, outputs=[logits_variance, softmax_output])


def load_bayesian_model(checkpoint, monte_carlo_simulations=100, classes=10):
	get_custom_objects().update({"bayesian_categorical_crossentropy_internal": bayesian_categorical_crossentropy(monte_carlo_simulations, classes)})
	return load_model(checkpoint)


def load_epistemic_uncertainty_model(checkpoint, epistemic_monte_carlo_simulations):
	model = load_bayesian_model(checkpoint)
	inpt = Input(shape=(model.input_shape[1:]))
	x = RepeatVector(epistemic_monte_carlo_simulations)(inpt)
	# Keras TimeDistributed can only handle a single output from a model :(
	# and we technically only need the softmax outputs.
	hacked_model = Model(inputs=model.inputs, outputs=model.outputs[1])
	x = TimeDistributed(hacked_model, name='epistemic_monte_carlo')(x)
	# predictive probabilties for each class
	softmax_mean = TimeDistributedMean(name='epistemic_softmax_mean')(x)
	variance = PredictiveEntropy(name='epistemic_variance')(softmax_mean)
	epistemic_model = Model(inputs=inpt, outputs=[variance, softmax_mean])

	return epistemic_model


def load_full_epistemic_uncertainty_model(encoder, input_shape, checkpoint, epistemic_monte_carlo_simulations):
	encoder_model = create_encoder_model(encoder, input_shape)
	bayesian_model = load_epistemic_uncertainty_model(checkpoint, epistemic_monte_carlo_simulations)
	outputs = bayesian_model(encoder_model.outputs)

	return Model(inputs=encoder_model.inputs, outputs=outputs)


def create_bayesian_model(encoder, input_shape, output_classes):
	encoder_model = create_encoder_model(encoder, input_shape)
	input_tensor = Input(shape=encoder_model.output_shape[1:])
	x = BatchNormalization(name='post_encoder')(input_tensor)
	x = Dropout(0.5)(x)
	x = Dense(500, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.5)(x)
	x = Dense(100, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.5)(x)

	logits = Dense(output_classes)(x)
	variance_pre = Dense(1)(x)
	variance = Activation('softplus', name='variance')(variance_pre)
	logits_variance = concatenate([logits, variance], name='logits_variance')
	softmax_output = Activation('softmax', name='softmax_output')(logits)

	model = Model(inputs=input_tensor, outputs=[logits_variance,softmax_output])

	return model


def create_encoder_model(encoder, input_shape):
	input_tensor = Input(shape=input_shape)

	if encoder == 'resnet50':
		base_model = ResNet50(include_top=False, input_tensor=input_tensor)
	else:
		raise ValueError('Unexpected encoder model ' + encoder + ".")

	# freeze encoder layers to prevent over fitting
	for layer in base_model.layers:
		layer.trainable = False
	
	output_tensor = Flatten()(base_model.output)

	model = Model(inputs=input_tensor, outputs=output_tensor)
	return model

def encoder_min_input_size(encoder):
	if encoder == 'resnet50':
		return (197, 197)
	else:
		raise ValueError('Unexpected encoder model ' + encoder + ".")



def extract_last_row(shape, dtype=None):
	extractor = np.zeros(shape)
	extractor[-1][-1] = 0
	return K.constant(extractor)

def drop_last_row(shape, dtype=None):
	extractor = np.zeros(shape)
	for i in range(np.min(shape)):
		extractor[i][i] = 1
	return K.constant(extractor)

def extract_variance(prev_layer):
	layer = Dense(1, name='extract_variance', kernel_initializer=extract_last_row)
	layer.trainable = False
	return layer(prev_layer)

def extract_logits(prev_layer, output_classes):
	layer = Dense(output_classes, name='extract_logits', kernel_initializer=drop_last_row)
	layer.trainable = False
	return layer(prev_layer)
