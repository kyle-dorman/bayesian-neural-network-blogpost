#!/bin/python 

from keras.applications.resnet50 import ResNet50
# from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import Dense, Input, Flatten, Dropout, Activation, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate

def create_model(input_shape, output_classes):

	inpt = Input(shape=input_shape)
	encoder = ResNet50(include_top=False, input_tensor=inpt)
	x = Flatten()(encoder.output)
	x = BatchNormalization(name='post_encoder')(x)
	x = Dropout(0.5)(x)
	x = Dense(500, activation='relu')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.5)(x)
	logits = Dense(output_classes, name='logits')(x)
	variance = Dense(1, name='variance')(x)
	logits_variance = concatenate([logits, variance], name='logits_variance')
	softmax_output = Activation('softmax', name='softmax_output')(logits)

	model = Model(inputs=inpt, outputs=[logits_variance,softmax_output])

	return model
