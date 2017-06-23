#!/bin/python 

from keras.applications.resnet50 import ResNet50
# from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model, load_model
from keras.layers import Dense, Input, Flatten, Dropout, Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate

def load_full_model(model_name, checkpoint, input_shape):
	encoder = create_encoder_model(model_name, input_shape)
	baysean_model = load_baysean_model(checkpoint)
	outputs = baysean_model(encoder.outputs)
	# hack to rename outputs
	logits_variance = Lambda(lambda x: x, name='logits_variance')(outputs[0])
	softmax_output = Lambda(lambda x: x, name='softmax_output')(outputs[1])

	return Model(inputs=encoder.inputs, outputs=[logits_variance, softmax_output])

def load_baysean_model(checkpoint):
	return load_model(checkpoint)

def create_baysean_model(model_name, input_shape, output_classes):
	encoder = create_encoder_model(model_name, input_shape)
	input_tensor = Input(shape=encoder.output_shape[1:])
	x = BatchNormalization(name='post_encoder')(input_tensor)
	x = Dropout(0.5)(x)
	x = Dense(500, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.5)(x)
	logits = Dense(output_classes, name='logits')(x)
	variance_pre = Dense(1, name='variance_pre')(x)
	variance = Activation('softplus', name='variance')(variance_pre)
	logits_variance = concatenate([logits, variance], name='logits_variance')
	softmax_output = Activation('softmax', name='softmax_output')(logits)

	model = Model(inputs=input_tensor, outputs=[logits_variance,softmax_output])

	return model

def create_encoder_model(model_name, input_shape):
	input_tensor = Input(shape=input_shape)

	if model_name == 'resnet50':
		base_model = ResNet50(include_top=False, input_tensor=input_tensor)
	else:
		raise ValueError('Unexpected encoder model ' + model_name + ".")

	# freeze encoder layers to prevent over fitting
	for layer in base_model.layers:
		layer.trainable = False
	
	output_tensor = Flatten()(base_model.output)

	model = Model(inputs=input_tensor, outputs=output_tensor)
	return model
		
def encoder_min_input_size(model_name):
	if model_name == 'resnet50':
		return (197, 197)
	else:
		raise ValueError('Unexpected encoder model ' + model_name + ".")
