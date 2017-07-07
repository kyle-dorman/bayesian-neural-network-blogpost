#!/bin/python 

import numpy as np
from keras import backend as K
from tensorflow.contrib import distributions

# model - the trained classifier(C classes) 
#					where the last layer applies softmax
# X_data - a list of input data(size N)
# T - the number of monte carlo simulations to run
def montecarlo_prediction(model, X_data, T):
	# shape: (T, N, C)
	predictions = np.array([model.predict(X_data) for _ in range(T)])

	# shape: (N, C)
	prediction_means = np.mean(predictions, axis=0)
	
	# shape: (N)
	prediction_variances = np.apply_along_axis(predictive_entropy, axis=1, arr=prediction_means)
	return (prediction_means, prediction_variances)

# prob - mean probability for each class(C)
def predictive_entropy(prob):
	return -np.sum(np.log(prob) * prob)


# standard regression RMSE loss function
# N data points
# true - true values. Shape: (N)
# pred - predicted values. Shape: (N)
# returns - losses. Shape: (N)
def loss(true, pred):
	return np.mean(np.square(pred - true))

# Bayesian regression loss function
# N data points
# true - true values. Shape: (N)
# pred - predicted values (mean, log(variance)). Shape: (N, 2)
# returns - losses. Shape: (N)
def loss_with_uncertainty(true, pred):
	return np.mean((pred[:, :, 0] - true)**2. * np.exp(-pred[:, :, 1]) + pred[:, :, 1])



# standard categorical cross entropy
# N data points, C classes
# true - true values. Shape: (N, C)
# pred - predicted values. Shape: (N, C)
# returns - loss (N)
def categorical_cross_entropy(true, pred):
	return np.sum(true * np.log(pred), axis=1)

# Bayesian categorical cross entropy.
# N data points, C classes, T monte carlo simulations
# true - true values. Shape: (N, C)
# pred_var - predicted logit values and variance. Shape: (N, C + 1)
# returns - loss (N,)
def bayesian_categorical_crossentropy(T, num_classes):
  def bayesian_categorical_crossentropy_internal(true, pred_var):
    # shape: (N, C)
    std = true * K.sqrt(pred_var[:, num_classes:])
    # shape: (N,)
    variance = pred_var[:, num_classes] + K.ones_like(pred_var[:, num_classes])
    # shape: (N, C)
    pred = pred_var[:, 0:num_classes]
    # shape: (T,)
    iterable = K.variable(np.ones(T))
    dist = distributions.Normal(loc=K.zeros_like(std), scale=std)
    # shape: (N,)
    undistorted_loss = K.categorical_crossentropy(pred, true, from_logits=True)
    monte_carlo_results = K.map_fn(gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss), iterable, name='monte_carlo_results')
    return K.mean(monte_carlo_results, axis=0) * orig_loss + orig_loss + K.pow(K.log(variance), 3)
  return bayesian_categorical_crossentropy_internal

# for a single monte carlo simulation, 
#   calculate categorical_crossentropy of 
#   predicted logit values plus gaussian 
#   noise vs true values.
# true - true values. Shape: (N, C)
# pred - predicted logit values. Shape: (N, C)
# dist - normal distribution to sample from. Shape: (N, C)
# undistorted_loss - the crossentropy loss without variance distortion. Shape: (N,)
# returns - total differences for all classes (N,)
def gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss):
  def map_fn(i):
    distorted_loss = K.categorical_crossentropy(pred + dist.sample(1), true, from_logits=True)
    diff = undistorted_loss - distorted_loss
    return -K.elu(diff)
  return map_fn

    
class MonteCarloTestModel:
	def __init__(self, C):
		self.C = C

	def predict(self, X_data):
		return np.array([self._predict(data) for data in X_data])

	def _predict(self, data):
		return self.softmax([i for i in range(self.C)])

	def softmax(self, predictions):
		vals = np.exp(predictions)
		return vals / np.sum(vals)




