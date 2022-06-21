import numpy as np


def evaluate_dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = np.ndarray.flatten(y_true)
    y_pred_f = np.ndarray.flatten(y_pred)
    intersection = np.sum(y_true_f*y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def evaluate_dice_coef_loss(y_true, y_pred, smooth=1.):
    return 1- evaluate_dice_coefficient(y_true, y_pred)

def evaluate_mean_iou(y_true, y_pred):
	y_true_f = np.ndarray.flatten(y_true)
	y_pred_f = np.ndarray.flatten(y_pred)
	intersection = np.sum(y_true_f * y_pred_f)
	# calculate the |union| (OR) of the labels
	union = np.sum(y_true) + np.sum(y_pred) - intersection
	# avoid divide by zero - if the union is zero, return 1
	# otherwise, return the intersection over union
	if union == 0:
		iou = 1
	else:
		iou = intersection/union
	return iou