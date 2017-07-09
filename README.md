# Including uncertainty in classification using Bayesian deep learning

[//]: # (Image References)

[remoteimage1]: http://mlg.eng.cam.ac.uk/yarin/blog_images/gp_net.jpg "Bayesian deep learning"
[remoteimage2]: https://alexgkendall.github.io/assets/images/blog_uncertainty/uncertainty_types.jpg "Segmentation uncertainty"

[image1]: ./blog_images/aleatoric_variance_loss_function_analysis.png "Aleatoric variance vs loss for different incorrect logit values"
[image2]: ./blog_images/catdog.png "Ambiguity example"
[image3]: ./blog_images/example_images.png "Example Cifar10 images"
[image4]: ./blog_images/gammas.png "Example image with different gamma values"
[image5]: ./blog_images/max_aleatoric_uncertainty_test.png "Max Aleatoric Uncertainty"
[image6]: ./blog_images/max_epistemic_uncertainty_test.png "Max Epistemic Uncertainty"
[image7]: ./blog_images/softmax_categorical_crossentropy_v_logit_difference.png "Softmax categorical crossentropy vs. logit difference"
[image8]: ./blog_images/test_stats.png "Stats"
[image9]: ./blog_images/test_first_second_rest_stats.png "Stats by correct label logit position"
[image10]: ./blog_images/change_logit_loss_analysis.png "Change in logit loss"
[image11]: ./blog_images/catdog_just_dog.png "Just dog"
[image12]: ./blog_images/catdog_just_cat.png "Just cat"
[image13]: ./blog_images/augmented_vs_original_uncertainty.png "Uncertainty: augmented vs original images"

### Intro
In this blog post I will go over how to train a neural network classifier using [Keras](https://keras.io/) and [tensorflow](https://www.tensorflow.org/) to not only predict an outcome variable but also how uncertain the model is about its prediction using Bayesian deep learning techniques. I will first explain what uncertainty is and why it is important. I will then explain what Bayesian deep learning is and cover two techniques for including uncertainty in a deep learning model. To demonstrate my results I will use Keras to train dense layers over a frozen [ResNet50](https://arxiv.org/abs/1512.03385) encoder on the [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. Using less than 200 epochs and a custom loss object, I was able to train my model to score 86.4% on the training set. Lastly, I will explore the uncertainty predictions of my model and suggest next steps.

![alt image][image3]

Figure 1: example of each class in cifar10

### Acknowledgments
The code in the [github repo](https://github.com/kyle-dorman/bayesian-neural-network-blogpost) is original work but the bulk of the text and all of my knowledge about Bayesian deep learning comes from a few blog posts([here](http://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/) and [here](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html)) as well as a few papers from the Cambridge machine learning group. I want to say thank you this group for all their amazing blog posts and papers that make this difficult concept easier to understand. 

### What is [uncertainty](https://en.wikipedia.org/wiki/Uncertainty)
Uncertainty is the state of having limited knowledge where it is impossible to exactly describe the existing state, a future outcome, or more than one possible outcome. As it pertains to deep learning and classification, uncertainty also includes ambiguity; uncertainty about human definitions and concepts, not an objective fact of nature.

![alt image][image2]

Figure 2: an example of ambiguity. What should the model predict?

### Types of uncertainty
There are a few different types of uncertainty and I will only cover two important types in this post.

#### Aleatoric uncertainty
Measures what you can't understand from the data. It can be explained away with the ability to observe all explanatory variables with increased precision. Think of aleatoric uncertainty as sensing uncertainty. There are actually two types of aleatoric uncertainty, heteroscedastic and homoscedastic, but I am only covering heteroscedastic uncertainty in this post. Homoscedastic is covered more in depth in some of the blog post mentioned above. 

Concrete examples of aleatoric uncertainty in stereo imagery are occlusions(parts of the scene a camera can't see), lack of visual features (i.e a blank wall), or over/under exposed areas (glare & shading).

#### Epistemic uncertainty
Measures what your model doesn't know due to lack of training data. It can be explained away with increased training data. Think of epistemic uncertainty as model uncertainty.

If you trained an image classification model to predict the ten cifar10 classes but the airplane class images in your training set where all of airplanes in the sky, your model might predict high epistemic uncertainty when shown an image of an airplane on the ground because this is not a situation it has been trained for. 

An easy way to observe epistemic uncertainty in action is to train a model on 25% of your dataset and train a model on the entire dataset. A model trained on only 25% of the dataset will have higher average epistemic uncertainty than the model trained on the entire dataset because its seen fewer examples. 

### Why is uncertainty important
Whether we use deep learning or other techniques, in machine learning we are trying to create approximate representations of the real world. Popular deep learning models created today produce a point estimate but not an uncertainty value. Understanding if your model is under-confident or falsely over-confident can help you reason about your model and your dataset.

Note: In a classification problem, the softmax output does give you a probability, but this is not the same as uncertainty. Softmax probability is a probability distribution over your possible outcomes. It explains how confident your model is relative to the other options which is different than your model being confident. 

The two types of uncertainty explained above are import for different reasons. Aleatoric uncertainty is important in cases where parts of the observation space have higher noise levels than others. One particular example where aleatoric uncertainty comes to mind, is the first fatality involving a self driving car. Tesla has said that during this incident its camera failed to recognize the white truck against a bright sky. An image segmentation classifier that is able to predict aleatoric uncertainty could have been trained to recognize this difficult image space and placed a high uncertainty there. Rather just using a point estimation, the uncertainty, combine with the car's radar data could have helped the car better understand what was in front of it.

Epistemic uncertainty is important in safety critical applications because it helps identify situations that are different from training data. Epistemic uncertainty is also helpful for exploring your dataset. Epistemic uncertainty would have been helpful with [this](https://neil.fraser.name/writing/tank/) particular neural network mishap from the 1980s. Researchers trained a neural network to recognize tanks hidden in trees vs. treeswithout tanks. After training, the network preformed incredibly well on the training set and the test set! The only problem was all of the images of the tanks were taken on cloudy days and images without tanks were taken on a sunny day. Whoops. 

Another example where epistemic uncertainty could be useful is in the now famous Not Hotdog app. In practice the model preforms very well but it seems the network was never trained on 'not hotdog' images that included ketchup on the item in the image.

<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">If there&#39;s ketchup, it&#39;s a hotdog <a href="https://twitter.com/FunnyAsianDude">@FunnyAsianDude</a> <a href="https://twitter.com/hashtag/nothotdog?src=hash">#nothotdog</a> <a href="https://twitter.com/hashtag/NotHotdogchallenge?src=hash">#NotHotdogchallenge</a> <a href="https://t.co/ZOQPqChADU">pic.twitter.com/ZOQPqChADU</a></p>&mdash; David (@david__kha) <a href="https://twitter.com/david__kha/status/865093285886304256">May 18, 2017</a></blockquote>
<script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>

This ambitious (and possibly bored) 'tweeter' identified a case where the Not Hotdog model would probably exhibit large epistemic uncertainty.

Uncertainty predictions in deep learning models is also important in robotics. I am currently enrolled in the Udacity self driving car nanodegree, and have been learning about techniques cars/robots used to recognize and track objects around then. Self driving cars use a powerful technique called [Kalman filters](https://en.wikipedia.org/wiki/Kalman_filter) to track objects. Kalman filters combine a series of measurement data containing statistical noise and produce estimates that tend to be more accurate than any single measurement. Traditional deep learning models are not able to contribute to Kalman filters because they only predict an outcome and do not include an uncertainty term. Training models that are able to predict uncertainty would, in theory, allow them to contribute to Kalman filter tracking.

### Bayesian deep learning
The idea of including uncertainty in neural networks was proposed as early as [1991](http://papers.nips.cc/paper/419-transforming-neural-net-output-levels-to-probability-distributions.pdf). Instead of just having layers with weight parameters and biases, imagine placing a distribution over each weight parameter in your model and you begin to understand Bayesian deep learning. Because Bayesian deep learning models require more parameters to optimize, they are difficult to work with and have not been used very often. More recently, Bayesian deep learning has become popular again and new techniques are being developed to include uncertainty in a model with less model complexity.

![alt image][remoteimage1]

Visualizing a Bayesian deep learning model. 

### Calculating uncertainty in deep learning classification models
Because aleatoric and epistemic uncertainty are different, they are calculated differently. 

#### Calculating aleatoric uncertainty
Aleatoric uncertainty is a function of the input data. As such, a classification model can learn to predict aleatoric uncertainty by using a modified loss function. First, instead of only predicting the softmax values, the Bayesian classification model will have two outputs, the softmax values and the input variance. Because the model doesn't have variance labels to learn from, this is an example of unsupervised learning. Below is the standard categorical cross entropy loss function and a function to calculate the Bayesian categorical cross entropy loss.

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
# returns - loss (N,)
def bayesian_categorical_crossentropy(T, num_classes):
  def bayesian_categorical_crossentropy_internal(true, pred_var):
    # shape: (N,)
    std = K.sqrt(pred_var[:, num_classes:])
    # shape: (N,)
    variance = pred_var[:, num_classes]
    variance_depressor = K.exp(variance) - K.ones_like(variance)
    # shape: (N, C)
    pred = pred_var[:, 0:num_classes]
    # shape: (N,)
    undistorted_loss = K.categorical_crossentropy(pred, true, from_logits=True)
    # shape: (T,)
    iterable = K.variable(np.ones(T))
    dist = distributions.Normal(loc=K.zeros_like(std), scale=std)
    monte_carlo_results = K.map_fn(gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes), iterable, name='monte_carlo_results')
    
    variance_loss = K.mean(monte_carlo_results, axis=0) * undistorted_loss
    
    return variance_loss + undistorted_loss + variance_depressor
  
  return bayesian_categorical_crossentropy_internal

# for a single monte carlo simulation, 
#   calculate categorical_crossentropy of 
#   predicted logit values plus gaussian 
#   noise vs true values.
# true - true values. Shape: (N, C)
# pred - predicted logit values. Shape: (N, C)
# dist - normal distribution to sample from. Shape: (N, C)
# undistorted_loss - the crossentropy loss without variance distortion. Shape: (N,)
# num_classes - the number of classes. C
# returns - total differences for all classes (N,)
def gaussian_categorical_crossentropy(true, pred, dist, undistorted_loss, num_classes):
  def map_fn(i):
    std_samples = K.transpose(dist.sample(num_classes))
    distorted_loss = K.categorical_crossentropy(pred + std_samples, true, from_logits=True)
    diff = undistorted_loss - distorted_loss
    return -K.elu(diff)
  return map_fn
```

The loss function I created is based on the loss function in [this](https://arxiv.org/pdf/1703.04977.pdf) paper. In the paper, the loss function creates a normal distribution with zero mean and the predicted variance. It distorts the predicted logit values by sampling from the distribution and computes the softmax categorical cross entropy using the distorted predictions. It does this T times and then takes the average of the T samples as the loss. 

Taking the categorical cross entropy of the distorted logits should ideally result in a few interesting properties. First, when the predicted logit value is much larger than any other logit value(right half of Figure 3), increasing the variance should only increase the loss. This is true because the derivative is negative on the right half of the graph. The minimum loss should be close to 0 in this case. When the wrong logit(w) is much larger than the right logit(r) and the variance is 0, the loss should be ~`w-r`. You can see this is on the right half of Figure 3. To enable the model to learn aleatoric uncertainty, when the wrong logit value is greater than the correct logit value the loss function should be minimized for a variance value greater than 0. 

![alt image][image7]

Figure 3: Softmax categorical cross entropy vs. logit difference for binary classification

I was able to use the loss function suggested in the paper to decrease the loss when the wrong logit value is greater than the right logit value by increasing the variance, but the loss difference was extremely small (<0.1). During training, my model had a hard time picking up on this slight difference and the aleatoric variance predictions from my model did not make sense. I believe this happens because the slope of Figure 3 on the left half of the graph is ~ -1. Distorting the logits using a normal distribution and applying softmax cross entropy results in another normal distribution and the mean of the normal distribution is 0. Ideally the model is be rewarded in situations where it increases the right logit value above the wrong logit value and only slightly discouraged for decreasing the right logit further.

To make the model easier to train, I wanted to create a more significant loss change as the variance increases. Just like in the paper, my loss function above does compute the categorical cross entropy for T samples of the distorted logits using a normal distribution with a mean of 0 and the predicted variance. But from there my loss function deviates from the paper. 

I applied the [elu](http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf) activation function to the change in categorical cross entropy, `undistorted_loss - distorted_loss`. Because elu is a non-linear function around 0, it shifts the mean of the normal distribution away from zero for the left half of Figure 3 but maintains the mean for the right half of Figure 3. 

![alt image][image10]

Figure 4: Average change in loss & distorted average change in loss.

In Figure 4 wrong_logit_loss_distribution corresponds to a point on the left half of Figure 3 and right_logit_loss_distribution corresponds to a point on the right half of Figure 4. You can see that the distribution of outcomes from the wrong logit case, looks similar to the normal distribution and the right case is mostly small values near zero. After applying `-elu` to the change in loss, the mean of the wrong_logit_loss_distribution becomes much larger. The mean of the right_logit_loss_distribution stays about the same. I call the mean of the lower graphs in Figure 4 the distorted average change in loss.

I then scaled the distorted average change in loss by the undistorted categorical cross entropy. This is done because the distorted average change in loss for the wrong logit case is about the same for all logit differences greater than three(because the derivative of the line is 0). To ensure the loss is greater than zero, I add the undistorted categorical cross entropy. And to ensure the variance that minimizes the loss is less than infinity, I add the exponential of the variance.

![alt image][image1]

Figure 5: Aleatoric variance vs loss for different incorrect logit values

These are the results of calculating the above loss function for binary classification example where the true logit value is held constant at 1.0 and the wrong logit value changes for each line. I run 10000 Monte Carlo simulations to attempt to create smooth lines. When the wrong logit value is less than 1.0 (and thus less than the correct logit value), the minimum variance is 0.0. As the wrong logit value increases the variance that minimizes the loss increases. 

#### Calculating epistemic uncertainty
Epistemic uncertainty takes more code to model but in my experience is easier to work with. One way of modeling epistemic uncertainty is using Monte Carlo dropout sampling (a type of variational inference) at test time. For a full explanation of why dropout can model uncertainty check out [this](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html) blog and [this](https://arxiv.org/pdf/1703.04977.pdf) paper. In practice, this means including dropout in your model and running your model multiple times with dropout turned on at test time to create a distribution of outcomes and then calculating the predictive entropy (the average amount of information contained in the predictive distribution). 

To understand using dropout to calculate epistemic uncertainty, think about splitting the cat-dog image above in half vertically. 

![alt image][image11] ![alt image][image12]

If you saw the left half, you would predict dog. If you saw the right half you would predict cat. A perfect 50-50 split. This image would high epistemic uncertainty because the image exhibits features that the model (you) associates with both a cat class and a dog class. 

Below are two ways of calculating epistemic uncertainty. They do the exact same thing, but the first is simpler and only uses numpy. The second, uses additional Keras layers (and gets GPU acceleration) to make the predictions.

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
# input of shape (None, ...) returns output of same size.
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
	# predictive probabilities for each class
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

Note: Epistemic uncertainty is not used to train the model. It is only calculated at test time (but during a training phase) when evaluating test/real world examples. Where as aleatoric uncertainty is part of the training process.

### Training
Besides the training code above, predicting uncertainty doesn't require much additional code beyond the normal code used to train a classifier. 

```python
def resnet50(input_shape):
	input_tensor = Input(shape=input_shape)
	base_model = ResNet50(include_top=False, input_tensor=input_tensor)
	# freeze encoder layers to prevent over fitting
	for layer in base_model.layers:
		layer.trainable = False

	output_tensor = Flatten()(base_model.output)
	return Model(inputs=input_tensor, outputs=output_tensor)
```
For this experiment, I used the frozen convolutional layers from Resnet50 with the weights for [ImageNet](http://www.image-net.org/) to encode the images. I initially attempted to train the model without freezing the convolutional layers but found the model quickly became over fit.

```python
def create_bayesian_model(encoder, input_shape, output_classes):
	encoder_model = resnet50(input_shape)
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
```

The trainable part of my model is 2 sets of `BatchNormalization`, `Dropout`, `Dense`, and `relu` layers on top of the ResNet50 output. The logits and variance are calculated using separate `Dense` layers. An important note is the variance layer applies a `softplus` activation function to ensure the model always predicts variance values greater than zero. I then recombine the logits and variance for the aleatoric loss function and calculate the softmax from the logits. 

```python
model.compile(
	optimizer=Adam(lr=1e-3, decay=0.001),
	loss={
	'logits_variance': bayesian_categorical_crossentropy(100, 10),
	'softmax_output': 'categorical_crossentropy'
	},
	metrics={'softmax_output': metrics.categorical_accuracy},
	loss_weights={'logits_variance': .2, 'softmax_output': 1.})
```

I trained the model using two losses, one is the aleatoric uncertainty loss function and the other is the standard categorical cross entropy function. This allows the last `Dense` layer which creates the logits to only learn how to produce better logit values while the `Dense` layer that creates the variance only learns about predicting variance. The two prior `Dense` layers will train on both of these losses. Because the aleatoric uncertainty loss function includes the categorical cross entropy function, I weighted it less than the real categorical cross entropy function. These weights could be tweaked more.

I used 100 Monte Carlo simulations for calculating the Bayesian loss function. It took about 70 seconds per epoch. I found increasing the number of Monte Carlo simulations from 100 to 1000 added about four minutes to each training epoch. 

I added augmented data to the training set by randomly applying a gamma value of 0.5 or 2.0 to decrease or increase the brightness of each image. In practice I found the cifar10 dataset did not have many images that would in theory exhibit high aleatoric uncertainty. By adding the lightness & darkness to images in the training set I am attempting to give the model more images with what should be high aleatoric uncertainty to learn from. 

![alt image][image4]
Image: Example image with gamma value distortion. 1.0 is no distortion

Predicting epistemic uncertainty does take a considerable amount of time compared to just computing the softmax output and the aleatoric uncertainty. It takes about 2-3 seconds on my Mac CPU for the fully connected layers to predict all 50000 classes for the training set but over five minutes for the epistemic uncertainty predictions. This isn't that surprising because epistemic uncertainty requires running Monte Carlo simulations on each image. I ran 100 simulations so I expected the epistemic uncertainty predictions to take ~250 seconds. 

Lastly, The [repo](https://github.com/kyle-dorman/bayesian-neural-network-blogpost) is set up to easily switch out the underlying encoder network and train models for other datasets in the future. Feel free to play with it if you want a deeper dive into training your own Bayesian neural network classifier. 

### Results
My model's categorical accuracy on the training data is 86.4%. This is not an amazing score by any means. I was able to product higher scores, above 93%, but only by sacrificing the accuracy of the aleatoric uncertainty. There are a few different hyper parameters I could play with to increase my score. The most likely is the weights of my two loss functions. I could also unfreeze the Resnet50 layers and train those as well. While getting better accuracy scores on this dataset is interesting, Bayesian deep learning is about both the predictions and the uncertainty estimates and I will spend the rest of the post evaluating the validity of the uncertainty predictions of my model.

![alt image][image8]

Figure 6: uncertainty mean and standard deviation for test set

The aleatoric uncertainty values tend to be much smaller than the epistemic uncertainty. These two values can't be compared directly on the same image. They can however be compared against the uncertainty values the model predicts for other images in this dataset. 

![alt image][image9]

Figure 7: Uncertainty to relative rank of correct logit value.

To further explore the uncertainty, I broke the test data into three groups based on the relative value of the correct logit. 'first' is all correct predictions (i.e logit value for the correct label was the largest value). 'second', the correct label is the second largest logit value. And 'rest' is all other relative values. 86.4% of samples are in the 'first' group, 8.7% are in the 'second' group, and 4.9% are in the 'rest' group. Figure 7 shows the mean and standard deviation of the aleatoric and epistemic uncertainty for the test set broken out by these three groups. As I was hoping, the epistemic and aleatoric uncertainties are correlated with the relative rank of the correct logit. This indicates the model is more likely to identify incorrect labels as situations it is unsure about. Additionally the model is predicting greater than zero uncertainty when the model's prediction is correct. I expected the model to exhibit this characteristic because the model can be uncertain even if it's prediction is correct .

![alt image][image5]

Images with highest aleatoric uncertainty

![alt image][image6]

Images with the highest epistemic uncertainty

Above are the images with the highest aleatoric and epistemic uncertainty. While it is interesting to look at, it is not exactly clear to me why these images images have high aleatoric or epistemic uncertainty. This is one downside to training an image classifier to produce uncertainty. The uncertainty for the entire image is reduced to a single value. It is often times much easier to understand uncertainty in an image segmentation model because it is easier to compare the results for each pixel in an image. 

![alt image][remoteimage2]

"Illustrating the difference between aleatoric and epistemic uncertainty for semantic segmentation. You can notice that aleatoric uncertainty captures object boundaries where labels are noisy. The bottom row shows a failure case of the segmentation model, when the model is unfamiliar with the footpath, and the corresponding increased epistemic uncertainty." From [this](http://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/) blog post.

If my model understands aleatoric uncertainty well, I should be able to input images with low contrast, high brightness/darkness or high occlusions and have my model predict larger aleatoric uncertainty. To test this theory, I applied a range of gamma values to my test images to increase/decrease the pixel intensity and predicted these augmented image labels.

![alt image][image13]

Figure 8: Left side: Images & uncertainties with gamma values applied. Right side: Images & uncertainties of original image.

The model's accuracy on the augmented images is 5.5%. This means the gamma images completely tricked my model. The model wasn't trained to score well on these large gamma distortions so that is to be expected. Figure 8 shows the predicted uncertainty for eight of the augmented images on the left and eight original uncertainties and images on the right. The first four images have the highest predicted aleatoric uncertainty of the augmented images and the last four had the lowest aleatoric uncertainty of the augmented images. I am excited to see that the model predicts higher aleatoric and epistemic uncertainties for each image! The aleatoric uncertainty should be larger because the averse lighting conditions make the images harder to understand and the epistemic uncertainty should be larger because the model has not been trained on images with larger gamma distortions. 

### Next Steps
This is just the tip the iceberg for Bayesian deep learning. One immediate next step is to continue playing with the loss weights and unfreezing the Resnet50 convolutional layers to see if I can get a better accuracy score without losing the uncertainty characteristics I've seen above. I could also try training a model on a dataset that has more images that exhibit high aleatoric uncertainty. One dataset good candidate dataset is the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) dataset which I've worked with in one of my Udacity projects. This dataset is specifically meant to make the classifier "cope with large variations in visual appearances due to illumination changes, partial occlusions, rotations, weather conditions". Sounds like aleatoric uncertainty to me. 

Another approach could be to find images that produce large uncertainties. Large epistemic uncertainties could mean types of images the dataset didn't cover well and the model isn't prepared to understand. To do this, I could use a library like [CleverHans](https://github.com/tensorflow/cleverhans) created by Ian Goodfellow. This library uses an adversarial network to help explore model vulnerabilities. For a Bayesian model, the uncertainties would also help reveal the model's vulnerabilities. Another library I am excited to explore is [Edward](http://edwardlib.org/), a Python library for probabilistic modeling, inference, and criticism. Edward supports creating network layers with probability distributions and makes it easy to perform variational inference. [This](https://alpha-i.co/blog/MNIST-for-ML-beginners-The-Bayesian-Way.html) blog post uses Edward to train a Bayesian neural network on the MNIST dataset. 

If you've made it this far, I am very impressed and appreciative. Hopefully this post has been a good overview of Bayesian deep learning and the possibilities it unlocks. 
