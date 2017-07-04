# Including uncertainty in classification using Bayesian deep learning

[//]: # (Image References)


[image1]: ./blog_images/example_images.png "Example Cifar10 iamges"
[image2]: ./blog_images/max_aleatoric_uncertainty_test.png "Max Aleatoric Uncertainty"
[image3]: ./blog_images/max_epistemic_uncertainty_test.png "Max Epistemic Uncertainty"
[image4]: ./blog_images/test_class_stats.png "Stats by class"
[image5]: ./blog_images/test_right_wrong_class.png "Stats by right wrong classification and class"
[image6]: ./blog_images/test_right_wrong_stats.png "Stats by right wrong"
[image7]: ./blog_images/test_stats.png "Stats"
[image8]: ./blog_images/catdog.png "cat-dog"
[image9]: ./blog_images/test_first_second_test_stats.png "Stats by correct label logit position"
[imag10]: ./blog_images/gammas.png "Example image with different gamma values"
[image11]: ./blog_images/gamma_prediction_score "Test prediction score using images distored by gamma value"
[image12]: ./blog_images/gamma_aleatoric_uncertainty.png "Aleatoric Uncertainty for different gamma values"
[image13]

### Intro
In this blog post I will go over how to train a neural network classifier using [Keras](https://keras.io/) and [tensorflow](https://www.tensorflow.org/) to not only predict an outcome variable but also how confident the model is in its prediction using Bayesian deep learning (or Bayesian neural networks). I will first explain what uncertainty is and why it is important. I will then cover two techniques for including uncertainty in a deep learning model. To demonstrate my results I will use Keras to train dense layers over a frozen [ResNet50](https://arxiv.org/abs/1512.03385) encoder on the [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. Using less than 200 epochs and a custom loss object, I was able to train my model to score 92.5% on the training set, good for 14th place in the cifar10 [standings](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130). Lastly, I wil do a deep dive into the uncertainty predictions of my model and suggest next steps.

![alt image][image1]
Figure 1: examples of each class in cifar10 (10 classes ;p)

### Thank you thank you, you're far to kind
The code in the [github repo](https://github.com/kyle-dorman/bayesian-neural-network-blogpost) is original work but the bulk of the text and all of my knowledge about Bayesain deep learning comes from a few blog posts([here](http://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/) and [here](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html)) as well as a few papers from the Cambridge machine learning group. Thank you them for all the amazing work and writing blog posts that make is difficult concept easy to understand. 

### What is [uncertainty](https://en.wikipedia.org/wiki/Uncertainty)
Uncertainty is the state of having limited knowledge where it is impossible to exactly describe the existing state, a future outcome, or more than one possible outcome. As it pertains to deep learning and classification, uncertainty also includes ambiguity; uncertainty about human definitions and concepts, not an objective fact of nature.

![alt image][image8]
Figure 2: an example of ambiguity. What should the model predict?

### Types of uncertainty
There are many types of uncertainty and I will only cover two important types in this post.

#### Aleatoric uncertainty
Measures what you can't understand from the data. It can be explained away with the ability to observe all explanatory variables with increased precision. Think of aleatoric uncertainty as sensing uncertainty (There are acutally two types of aleatoric uncertainty, heteroscedastic and homoscedastic, but I am only covering heteroscedastic uncertainty in this post).

For example, an image might have high aleatoric uncertanty due to occlusions (unable to see all of an object), lack of visual features, or over/under exposed areas.

#### Epistemic uncertainty
Measures what your model doesn't know due to lack of training data. It can be explained away with increased training data. Think of epistemic uncertainy as model uncertainty.

For example, if you trained an image classification model to predict the 10 cifar10 classes but the airplace class images where all of airplanes in the sky, your model would predict high epistemic uncertainty when shown an image of an airplace on the ground. An easy way to observe epistemic uncertainty in action is to train a model on 25% of your dataset and train a model trained on the entire dataset. A model trained on only 25% of the dataset will have higher average epistemic uncertainty than the model trained on the entire dataset because its seen less examples. 

### Why is uncertainty important
Whether we use deep learning or other techniques, in machine learning we are trying to create aproximate represntations of the real world. Popular deep learning models created today produce a point estimate/classification but not an uncertainty value. Understanding if your model is under-confident or falsely over-confident can help you reason about your model and your dataset.

Note: In a classification problem, the softmax output does give you a probablility, but this is not the same as uncertainty. Softmax probability is just a probablity distribution over your K possible outcomes. It explains how confident your model is relative to the other options. 

The two types of uncertainty explained above are import for different reasons. Aleatoric uncertainty is important in cases where parts of the obervation space have highter noise levels than others. One particular example where aleatoric uncertainty comes to mind, is the first fatality involving a self driving car. In this incident, Tesla has said its camera failed to recognize the white truck against a bright sky. An image segmentation classifier with aleatoric uncertainty predictions would probably have had predicted high uncertainty for two hard to distingish shapes/colors next to each other. This uncertainty combine with the car's radar data could have helped the car understand somethng was infront of it.

Epistemic uncertainty is important in safety critical applications because it is used to understand situations that are different from training data. Epistemic uncertainty is also helpful for exploring your dataset. Finding an image in a classification model with high epistemic uncertainty normally means there is somethng about that image that is particularly unique compared to the rest of your dataset. Epistemic uncertainty would have been particularly helpful with [this](https://neil.fraser.name/writing/tank/) particular neural network mishap from the 1980s. 

Uncertainty in deep learning models is also important in robotics. I am currently enrolled in the Udacity self driving car nanodegree, and have been learning about techniques cars/robots used to reccognize and track objects around then. Self driving cars use a powerful technique called [Kalman filters](https://en.wikipedia.org/wiki/Kalman_filter) to track objects. Kalman filters combine a series of measurement data containing statistical noise and produces estimates that tend to be more accurate than any single measurement. Traditional deep learning models (image segmentation for example) are not able to contribute to Kalman filters because they only predict an outcome and do not include an uncertainty term. Training models that are able to predict uncertainty would, in theory, allow them to contribute to Kalman filter tracking.

### Bayesian deep learning
Bayesian deep learning, including uncertainty in neural networks, was proposed as early as [1991](http://papers.nips.cc/paper/419-transforming-neural-net-output-levels-to-probability-distributions.pdf). Instead of just having layers with weight parameters and biases, imagine placing a distribution over each weight parameter in your model and you begin to understand Bayesian deep learning. Rather than just computing values as imformation is moved through the network, the netwrok would also say how confident it was in that particular value. Beacuse Bayesain deep learning models require more parameters to optimize, they are difficult to work with and have not been used very often. More recently, Bayesian deep learning has become popular again and new techniques are being developed to include uncertainty in a model with less model complexity.

### Calculating uncertainty in deep learning
Because aleatoric and epistemic uncertainty are different, they are calculated differently. 

Aleatoric uncertainty is a function of the input data so a model can learn to predict it by updating its loss function. Instead of only outputing the softmax values, the bayesian model will now have two outputs, the softmax values and the input variance. This is an example of unsupervised learning as we don't have uncertainty values to learn from. Below is the standard categorical cross entropy loss function and a set of functions to calculate the Bayesian categorical cross entropy loss (multiple functions is hopefully easier to understand).

```python
import numpy as np
from keras import backend as K
from tensorflow.contrib import distributions

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
# returns - loss (N)
def bayesian_categorical_crossentropy(T, num_classes):
	def bayesian_categorical_crossentropy_internal(true, pred_var):
		# shape: [N, 1]
		std_vals = K.sqrt(pred_var[:, num_classes:])
		# shape: [N, C]
		std = true * std_vals
		pred = pred_var[:, 0:num_classes]
		iterable = K.variable(np.ones(T))
		dist = distributions.Normal(loc=K.zeros_like(std), scale=std)
		# Shape: (T, N)
		monte_carlo_results = K.map_fn(gaussian_categorical_crossentropy(true, pred, dist), iterable, name='monte_carlo_results')
		return K.mean(monte_carlo_results, axis=0)
	return bayesian_categorical_crossentropy_internal

# for a single monte carlo simulation, 
#   calculate categorical_crossentropy of 
#   predicted logit values plus gaussian 
#   noise vs true values.
# true - true values. Shape: (N, C)
# pred - predicted logit values. Shape: (N, C)
# dist - normal distribution to sample from. Shape: (N, C)
# returns - categorical_crossentropy for each sample (N)
def gaussian_categorical_crossentropy(true, pred, dist):
	def map_fn(i):
		return K.categorical_crossentropy(pred + dist.sample(1), true, from_logits=True)
	return map_fn
```

The loss function is calculated using the model's predicted logits, the model's variance predictions, and the true classifications. Effectivly what the loss function does is corrupt the correct class predictions with Gaussian noise from the variance and apply a softmax function to the corrupted predictions. The loss functions uses a Monte Carlo simulation to sample the Gaussian distribution around the prediction. 



This loss function has a few desierable charicteristics. When the model gives the correct class a high logit value (relative to the other logit values) and a low variance, the loss from the Monte Carlo simulation will be close to 0. This is the ideal case when the model correctly classifies the input and has a high confidence in its answer. There are two interesting cases when the model predicts the wrong class (i.e. a different class has a higher logit value than the correct class). If the correct class has a low variance (the model is incorrectly confident that it should predict a non-correct class) the loss will be penalised by ~ the difference between the highest prediction and the correct predition. If instead the model has a high uncertainty for the correct class, the penalization will be decreased in cases where the correct class corrupted prediction is close to the highest prediction value. This means if the model predicts the wrong class but is unsure about its prediction, it is penalized less. 

Epistemic uncertainty is harder to model. One way of modeling it is using Monte Carlo dropout sampling (a type of variational inference) at test time. For a full explanation of why dropout can model uncertainty check out [this](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html) blog and [this](https://arxiv.org/pdf/1703.04977.pdf) paper. In practice this means including dropout in your model and running your model multiple times with dropout turned on at test time to create a distribution of outcomes and then calculating the predictive entropy (the average amount of information contained in the predictive distribution). 

Below are two ways of calculating epistemic uncertainty. The do the exact same thing, but the first is simplier and only uses numpy on the results of the predictions. The second, uses additional Keras layers (and possibly the GPU) to make the predictions.  

```python
# model - the trained classifier(C classes) 
#					where the last layer applies softmax
# X_data - a list of input data(size N)
# T - the number of monte carlo simulations to run
def montecarlo_prediction(model, X_data, T):
	# shape: (T, N, C)
	predictions = np.array([model.predict(X_data) for _ in range(T)])

	# shape: (N, C)
	prediction_probabilities = np.mean(predictions, axis=0)
	
	# shape: (N)
	prediction_variances = np.apply_along_axis(predictive_entropy, axis=1, arr=prediction_probabilities)
	return (prediction_probabilities, prediction_variances)

# prob - prediction probability for each class(C). Shape: (N, C)
# returns - Shape: (N)
def predictive_entropy(prob):
	return -1 * np.sum(np.log(prob) * prob, axis=1)
```

```python
from keras.models import Model
from keras.layers import Input, RepeatVector
from keras.engine.topology import Layer
from keras.layers.wrappers import TimeDistributed

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


def create_epistemic_uncertainty_model(checkpoint, epistemic_monte_carlo_simulations):
	model = load_saved_model(checkpoint)
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

# 1. Load the model
# 2. compile the model
# 3. Set learning phase to train
# 4. predict
def predict():
  model = create_epistemic_uncertainty_model('model.ckpt', 100)
  model.compile(...)

  # set learning phase to 1 so that Dropout is on. In keras master you can set this
	# on the TimeDistributed layer
	K.set_learning_phase(1)

	epistemic_predictions = model.predict(data)

```

Note: Epistemic uncertainty is not used to train the model. It is only calculated at test time (but during a training phase) when evaluating test/real world examples. Where as aleatoric is part of the training process.

### Code
Besides the training code above, predicting uncertainty doesn't require much additional code beyond the normal code used to train a classifier. For this experiment, I used the convolutional layers of ResNet50 frozen to the weights for [ImageNet](http://www.image-net.org/) and added 3 additional sets of `BatchNormalization`, `Dropout`, `Dense`, and `relu` layers on top of the ResNet50 output. I initally attempted to train the model without freezing the convolutional layers but found the model became overfit very quickly. After deciding to freeze the convolutional layers, I wrote a script that computes and saves the outputs of the encoder network which makes working with the smaller dataset extremly easy on my Mac. The variance and logits are predicted seperetly in the last layer so that it is easy to compute the softmax on just the logit values. An important note is the variance layer applies a `softplus` activation function to ensure the model always predicts variance values greater than zero. Predicting epistemic uncertainty does take a conisderably long amount of time. It takes about 2 seocnds on my Mac CPU for the fully connected layers part of the model to predict all 50000 classes for the training set but over five minutes for the epistemic uncertainty predictions. This isn't super suprising because epistemic uncertainty requires running T Monte Carlo simulations on each peice of data. I ran 100 simulations so I would have expected the epistemic uncertainty predictions to take ~200 seconds. The [repo](https://github.com/kyle-dorman/bayesian-neural-network-blogpost) is set up to easily switch out the underlying encoder network and train models for other datasets in the future. Feel free to play with it if you want a deeper dive. 

### Results
I was pleasantly surprised with the prediction results from the test data. 92.5% is very respectable given I spent no time messing around with the hyper parameters. Getting highly accurate scores on these datasets is covered in many different ways and Baysean deep learning is about both the predictions and uncertainty so I will spend the rest of the post analyzing the uncertainty predictions. 

![alt image][image7]
Figure 1: uncertainty mean and standard deviation for test set

The aleatoric uncertainty preditions by the model were a little dissapointing. The average uncertainty value is very low in comparison to the logit values. The average difference between the highest logit value and the second highest logit value was 6.9, meaning rarely was the aleatoric uncertainty helpful in reducing the loss. 

![alt image][image9]
Figure 2: uncertanty by relative value of the corrct label in the softmax output

To further explore the uncertainty, I broke the test data into the four groups based on the relative value of the corrct label in the softmax output. 'first' is all correct predictions (i.e softmax value for the correct label was the largest value). 'second' and 'third' are the correct label is the second and third highest softmax value respectivly. And 'rest' is all other values. 92.5% of samples are in the 'first' group, 5.4% are in the 'second' group, 1.3% are in the 'third' group and .8% are in the 'rest' group. Figure 2 shows the mean and standard deviation of the aleatoric and epistemic uncertainty for the test set broken out by the four groups. Interestingly, the aleatoric uncertainty does not seem to be cooralated with relative value of the corrct label in the softmax output. And in fact, the highest aleatoric uncertainty is when the correct label has the highest softmax value. Epistemic uncertainty is correlated with the relative value of the corrct label. 

![alt image][image2]
Figure 3: images with highest aleatoric uncertainty

One thing that jumps out to me is that none of these images seem particularly difficult to understand. The lighting is very good, the images hae fair amounts of contrast and there is little or no occlusions. At this point I am wondering if the aleatoric loss function was less successful in teaching my model to calculate loss than I had hoped. 

If my model understands aleatoric uncertainty well, I should be able to input images with low contrast, high brighness/darkness or high occlusions and have my model predict larger aleatoric uncertainty. To test this theory, I applied a range of gamma values to my test images to increase/decrease the pixel intensity and re-predicted my outputs.

![alt image][imag10]
Figure 4: example image with gamma value distortion. 1.0 is no distortion

![alt image][image11]
Figure 5: gamma value and test accuracy for different gamma values

Overall, the model is robust to most gamma value distortions. Only gamma values below 0.5 and above 1.9 seem to significantly effect the results. While this is great, I am more intersted in how the aleatoric uncertainty changes for different gamma values. 

![alt image][image12]
Figure 6: aleatoric uncertainty mean and uncertainty for test set with different gamma values applied to input images

If the model correctly understood how to predict aleatoric uncertainty, we would expact the average aleatoric uncertainty to increase as gamma moved away from 0. Looking at Figure 6, my model actaully produces the opposite affect. The average aleatoric uncertainty decreases away from a gamma value of 1.0. At this point I am convinced that my model's aleatoric uncertainty is not working as expected. 

Why is my model not "correctly" predicting aleatoric uncertainty? The first reason that comes to mind is that loss function doesn't behave as I want it to. To test this I modeled a three class prediction with varying logit values for the two incorrect classes while holding the true value constant at 1.0 with different uncertainty values. 















