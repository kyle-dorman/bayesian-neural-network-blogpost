# Including uncertainty in classification using Bayesian deep learning

### Intro
In this blog post I will go over how to train a neural network classifier to not only predict an outcome variable but also how confident the model is in its prediction using concepts known as Bayesian deep learning (or Bayesian neural networks). I will first explain what uncertainty is and why it is important before covering two techniques for including uncertainty in a deep learning model. To demonstrate my results I will use Keras to retrain ResNet50 on the MNIST database of handwritten digits. I will hopefully imporove or maintain the current ResNet50 score on this dataset while also identifying hard to understand digits based on the models uncertainty. 

### Thank you thank you, you're far to kind
The Keras code is original work but the bulk of the text and all of my knowledge about Bayesain deep learning comes from a few blog posts([here](http://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/) and [here](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html)) as well as a few papers from the Cambridge machine learning group.

### What is [uncertainty](https://en.wikipedia.org/wiki/Uncertainty)
A state of having limited knowledge where it is impossible to exactly describe the existing state, a future outcome, or more than one possible outcome. As it pertains to deep learning and classification, uncertainty also includes ambiguity; uncertainty about human definitions and concepts, not an objective fact of nature.

### Types of uncertainty
There are many types of uncertainty and I will only cover two important types here.

#### Epistemic uncertainty
Measures what your model doesn't know due to lack of training data. It can be explained away with increased training data. Think of epistemic uncertainy as model uncertainty.

For example, a classification model trained to reccognize handwritten letters with a training dataset of only printed letters would have high epistemic uncertainy classifying letters in cursive. 

#### Aleatoric uncertainty
Measures what you can't understand from the data. It can be explained away with the ability to observe all explanatory variables with increased precision. Think of aleatoric uncertainty as sensing uncertainty (There are acutally two types of aleatoric uncertainty, heteroscedastic and homoscedastic, but I am only covering heteroscedastic uncertainty in this post).

For example, an image might have high aleatoric uncertanty due to occlusions (unable to see all of an object), lack of visual features, or over/under exposed areas.

### Why is uncertainty important
Whether we use deep learning or other techniques, in machine learning we are trying to create aproximate represntations of the real world. Popular deep learning models produced today produce a point estimate/classification but not an uncertainty value(it is a misguided belief that using softmax to get probabilities is enough to obtain model uncertainty). Understanding if your model is under-confident or falsely over-confident can help you reason about your model and your dataset.

The two types of uncertainty explained above are import for different reasons. Epistemic uncertainty is important in safety critical applications because it is used to understand situations that are different from training data. Aleatoric uncertainty is important in cases where parts of the obervation space have highter noise levels than others. 

Uncertainty in deep learning models is also important in robotics. As I have learned in the Udacity self driving car nanodegree, self driving cars use a powerful technique called [Kalman](https://en.wikipedia.org/wiki/Kalman_filter) filters to track objects. Kalman filters combine a series of measurement data containing statistical noise and produces estimates that tend to be more accurate than any single measurement. Traditional deep learning models (image segmentation for example) are not able to contribute to Kalman filters because they only predict an outcome and do not include an uncertainty term.

### Bayesian deep learning
Bayesian deep learning, including uncertainty in neural networks, was proposed as early as [1991](http://papers.nips.cc/paper/419-transforming-neural-net-output-levels-to-probability-distributions.pdf). Imagine placing a distribution over each weight parameter in your model and you begin to understand Bayesian deep learning. Beacuse Bayesain deep learning models require more parameters to optimize, they are difficult to work with and have not been used very often. More recently, Bayesian deep learning has become popular again and new techniques are being developed to include uncertainty in a model with less model complexity.

### Calculating uncertainty in deep learning
Because aleatoric and epistemic uncertainty are different they are calculated differently. 

Aleatoric uncertainty is a function of the input data so a model can learn to predict it by updating its loss function. 
Instead of only outputing the softmax values, the bayesian model will now have three outputs, the softmax values, the logits, and the variances. Below is the standard categorical cross entropy loss function and a set of functions to calculate the Bayesian categorical cross entropy loss (multiple functions is hopefully easier to understand).

```python
import numpy as np
from keras import backend as K

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
# pred - predicted logit values. Shape: (N, C)
# variance - predicted variance. Shape: (N)
# returns - loss (N, C)
def bayesian_categorical_crossentropy(true, pred, variance, T):
	return np.mean([gaussian_categorical_crossentropy(true, pred, variance) for _ in range(T)], axis=0)

# for a single monte carlo simulation, 
#   calculate categorical_crossentropy of 
#   predicted logit values plus gaussian 
#   noise and true values.
# true - true values. Shape: (N, C)
# pred - predicted logit values. Shape: (N, C)
# variance - predicted variance. Shape: (N)
# returns - total differences for all classes (N)
def gaussian_categorical_crossentropy(true, pred, variance):
	std = np.ones_like(pred) * np.sqrt(variance)
	predictions = pred + np.random.normal(scale=std)
	return K.eval(K.categorical_crossentropy(predictions, true, from_logits=True))
```

The loss function is calculated using the model's predicted logits, the model's variance predictions, and the true classifications. Effectivly what the loss function does is corrupt the predictions with Gaussian noise with the variance and apply a softmax function to the corrupted predictions. The loss functions uses a Monte Carlo simulation to create a Gaussian distribution around the prediction. 

This loss function has a few desierable charicteristics. When the model gives the correct class a high logit value(relative to the other logit values) and a low variance, the loss from the Monte Carlo simulation will be close to 0. This is the ideal case when the model correctly classifies the input and has a high confidence in its answer. There are two interesting cases when the model predicts the wrong class (i.e. a different class has a higher logit value than the correct class). If the correct class has a low variance (the model is incorrectly confident that it should not predict the correct class) the loss will be penalised by ~ the difference between the highest prediction and the correct predition. If instead the model has a high uncertainty for the correct class, the penalization will be decreased in cases where the correct class corrupted prediction is close to the highest prediction value. This means if the model predicts the wrong class but is unsure about its prediction, it is penalized less. 

Epistemic uncertainty is harder to model. One way of modeling it is using Monte Carlo dropout sampling (a type of variational inference) at test time. For a full explanation of why dropout can model uncertainty check out [this](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html) blog and [this](https://arxiv.org/pdf/1703.04977.pdf) paper. In practice this means including dropout in your model and running your model multiple times with dropout turned on at test time to create a distribution of outcomes and then calculating the predictive entropy (the average amount of information contained in the predictive distribution). 

```python
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
	return -1 * np.sum(np.log(prob) * prob)
```
Note: Epistemic uncertainty is not used to train the model. It is only calculated at test time when evaluating test/real world examples. Where as aleatoric is part of the training process.

### Code
In this section I will cover re-training ResNet50 using the above Bayesean loss function and by reworking the last few layers using dropout to calculate epistemic.

















