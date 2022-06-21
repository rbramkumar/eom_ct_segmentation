import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow import keras as keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
#from keras.utils import plot_model
from tensorflow.keras import backend as K
import time
import tensorflow
from scipy.ndimage import distance_transform_edt as distance


#initializer = RandomNormal(mean=0., stddev=1.)
#initializer = GlorotNormal(seed=1)
initializer = GlorotUniform(seed=1)

############################### 2D UNET with Batch Norm ######################

def unet_2d_model_deep4(n_classes=1, im_sz=64, n_channels=1, n_filters_start=16, growth_factor=2, upconv=True):
	
	n_filters = n_filters_start
	inputs = Input((im_sz, im_sz, n_channels))

	inputs_n = GaussianNoise(10)(inputs)

	bn1   = BatchNormalization()(inputs_n)

	conv1 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False, data_format="channels_last")(bn1)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)
	pool1 = Dropout(rate = 0.2)(pool1)

	n_filters *= growth_factor
	conv2 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(pool1)
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)
	pool2 = Dropout(rate = 0.2)(pool2)

	n_filters *= growth_factor
	conv3 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(pool2)
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)
	pool3 = Dropout(rate = 0.2)(pool3)

	n_filters *= growth_factor
	conv4 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(pool3)
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)
	pool4 = Dropout(rate = 0.2)(pool4)
	
	
	n_filters *= growth_factor
	conv5 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(pool4)
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv5)
	
	
	n_filters //= growth_factor
	if upconv:
		up6 = concatenate([Conv2DTranspose(filters=n_filters, kernel_size=3, strides=2, padding='same', kernel_initializer = initializer, use_bias=False)(conv5), conv4])
	else:
		up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])
	conv6 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(up6)
	conv6 = BatchNormalization()(conv6)
	conv6 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv6)
	conv6 = Dropout(rate = 0.2)(conv6)

	n_filters //= growth_factor
	if upconv:
		up7 = concatenate([Conv2DTranspose(filters=n_filters, kernel_size=3, strides=2, padding='same', kernel_initializer = initializer, use_bias=False)(conv6), conv3])
	else:
		up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
	conv7 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(up7)
	conv7 = BatchNormalization()(conv7)
	conv7 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv7)
	conv7 = Dropout(rate = 0.2)(conv7)

	n_filters //= growth_factor
	if upconv:
		up8 = concatenate([Conv2DTranspose(filters=n_filters, kernel_size=3, strides=2, padding='same', kernel_initializer = initializer, use_bias=False)(conv7), conv2])
	else:
		up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
	conv8 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(up8)
	conv8 = BatchNormalization()(conv8)
	conv8 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv8)
	conv8 = Dropout(rate = 0.2)(conv8)

	n_filters //= growth_factor
	if upconv:
		up9 = concatenate([Conv2DTranspose(filters=n_filters, kernel_size=3, strides=2, padding='same', kernel_initializer = initializer, use_bias=False)(conv8), conv1])
	else:
		up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
	conv9 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(up9)
	conv9 = BatchNormalization()(conv9)
	conv9 = Conv2D(filters=n_filters, kernel_size=3, activation='relu', padding='same', kernel_initializer = initializer, use_bias=False)(conv9)
	conv9 = Dropout(rate = 0.2)(conv9)
	
	
	conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid', kernel_initializer = initializer, use_bias=False)(conv9)

	model = Model(inputs=inputs, outputs=conv10)
	
	return model


