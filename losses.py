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


def dice_coefficient(y_true, y_pred, smooth=1.):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
	return 1-dice_coefficient(y_true, y_pred)

##############  WEIGHTED BINARY CROSS ENTROPY LOSS  #######################

def create_weighted_binary_crossentropy(zero_weight, one_weight):

	def weighted_binary_crossentropy(y_true, y_pred):
		b_ce = K.binary_crossentropy(y_true, y_pred)
		weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
		weighted_b_ce = weight_vector * b_ce
		return K.mean(weighted_b_ce)
	return weighted_binary_crossentropy
# Set class weights
weighted_binary_crossentropy = create_weighted_binary_crossentropy(zero_weight=0.1, one_weight=0.9)

########################  DICE LOSS  #######################################

def dice_coefficient(y_true, y_pred, smooth=1.):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
	return 1-dice_coefficient(y_true, y_pred)



#####################  DICE + WBCE LOSS #####################################
def dice_wbce_loss(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)

	smooth=1.

	# Calculate wbce
	w = weighted_binary_crossentropy(y_true, y_pred)
	d = dice_coef_loss(y_true, y_pred)

	return w+d


smooth=1
	

##############  FOCAL LOSS (FOCAL TVERSKY) #######################
#https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
# Reference https://arxiv.org/abs/1810.07842
def tversky(y_true, y_pred):
	y_true_pos = K.flatten(y_true)
	y_pred_pos = K.flatten(y_pred)
	true_pos = K.sum(y_true_pos * y_pred_pos)
	false_neg = K.sum(y_true_pos * (1-y_pred_pos))
	false_pos = K.sum((1-y_true_pos)*y_pred_pos)
	alpha = 0.7
	return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
	return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
	pt_1 = tversky(y_true, y_pred)
	gamma = 0.75
	return K.pow((1-pt_1), gamma)

##############  JACCARD (INTERSECTION OVER UNION) #######################

def mean_iou(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true * y_pred)
	# calculate the |union| (OR) of the labels
	union = K.sum(y_true) + K.sum(y_pred) - intersection
	# avoid divide by zero - if the union is zero, return 1
	# otherwise, return the intersection over union
	return K.switch(K.equal(union, 0), 1.0, intersection / union)


# FROM https://github.com/LIVIAETS/boundary-loss/
# # Simple script which includes functions for calculating surface loss in keras
# ## See the related discussion: https://github.com/LIVIAETS/boundary-loss/issues/14
# REFERENCE: https://arxiv.org/pdf/1812.07032.pdf

from keras import backend as K
import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt as distance


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)


def surface_loss_keras(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)


# # Scheduler
# ### The following scheduler was proposed by @marcinkaczor
# ### https://github.com/LIVIAETS/boundary-loss/issues/14#issuecomment-547048076

class AlphaScheduler(Callback):
    def __init__(self, alpha, update_fn):
        self.alpha = alpha
        self.update_fn = update_fn
    def on_epoch_end(self, epoch, logs=None):
        updated_alpha = self.update_fn(K.get_value(self.alpha))
        K.set_value(self.alpha, updated_alpha)


alpha = K.variable(1, dtype='float32')

def gl_sl_wrapper(alpha):
    def gl_sl(y_true, y_pred):
        return alpha * dice_coef_loss(y_true, y_pred) + (1 - alpha) * surface_loss_keras(y_true, y_pred)
    return gl_sl


def update_alpha(value):
  return np.clip(value - 0.01, 0.01, 1)


# model.compile(loss=gl_sl_wrapper(alpha))
# history = model.fit_generator(
#   ...,
#   callbacks=AlphaScheduler(alpha, update_alpha)
# )