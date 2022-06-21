import numpy as np
import random
import os
from scipy.ndimage import rotate, zoom
import skimage.transform

def one_hot_encode_mask(mask):

	masks_encoded=[]

	for segment in range(8):
		mask_encode=np.zeros(mask.shape)
		mask_encode[mask==segment+1] = 1
		masks_encoded.append(mask_encode)
	
	masks_encoded = np.stack(masks_encoded, axis=-1)

	return masks_encoded

# Function to window image
def window_image(image, window_center, window_width):
	img_min = window_center - window_width // 2
	img_max = window_center + window_width // 2
	window_image = image.copy()
	window_image[window_image < img_min] = img_min
	window_image[window_image > img_max] = img_max
	
	return window_image

## HELPER FUNCTIONS FOR DATA GENERATOR

####################################### 2D data section #################################

#Find 1 random patch, from both img and mask
def get_2d_cor_rand_patch(img, mask, sz=64):
	"""
	:param img: ndarray with shape (x_sz, y_sz, z_sz)
	:param mask: ndarray with shape (x_sz, y_sz, z_sz)
	:param sz: size of random patch
	:return: patch with shape (x_sz_p, z_sz_p)
	"""
	
	# For coronal patch only x and z 
	assert len(img.shape) == 3 and img.shape[0] > sz and img.shape[1] > sz  and img.shape[0:2] == mask.shape[0:2]
	
	xc = random.randint(0, img.shape[0] - sz)
	yc = random.randint(0, img.shape[1] - sz) 
	zc = random.randint(0, img.shape[2])
		
	patch_img  =  img[xc:(xc + sz), yc:(yc + sz), zc]
	patch_mask = mask[xc:(xc + sz), yc:(yc + sz), zc]

	return patch_img, patch_mask


# Input X,Y and return random patches, with and without EYES
def get_2d_cor_patches(scans, masks, num_patches_required = 100, sz = 64, scale_factor=(100,101), intensity_factor=(100,101)):

	eye_muscle_threshold=500
	
	X = []
	y = []
	
	while len(X) < num_patches_required*0.8:

		i = random.randint(0,len(scans)-1)
		try:
			scale=np.random.randint(scale_factor[0],scale_factor[1])/100
			intensity=np.random.randint(intensity_factor[0],intensity_factor[1])/100
			
			scaled_patch_dim=int(scale*sz)
			
			random_X, random_Y = get_2d_cor_rand_patch(img = scans[i], mask = masks[i], sz = scaled_patch_dim)
			
			if np.count_nonzero(random_Y>0)>eye_muscle_threshold:
				random_X=zoom(random_X, zoom = sz/random_X.shape[0], order=3)
				random_Y=zoom(random_Y, zoom = sz/random_Y.shape[0], order=0)
				# k_index=[0,1,2,3]
				# k_rot = random.choice(k_index)
				# random_X = np.rot90(random_X, k=k_rot)
				# random_Y = np.rot90(random_Y, k=k_rot)
				X.append(random_X*intensity)
				y.append(random_Y)
		except: 
			pass
	
	while len(X) < num_patches_required:
		i = random.randint(0,len(scans)-1)
		try:
			scale=np.random.randint(scale_factor[0],scale_factor[1])/100
			intensity=np.random.randint(intensity_factor[0],intensity_factor[1])/100
			scaled_patch_dim=int(scale*sz)

			random_X, random_Y = get_2d_cor_rand_patch(img = scans[i], mask = masks[i], sz = scaled_patch_dim)
			random_X=zoom(random_X, zoom = sz/random_X.shape[0], order=3)
			random_Y=zoom(random_Y, zoom = sz/random_Y.shape[0], order=0)
			# k_index=[0,1,2,3]
			# k_rot = random.choice(k_index)
			# random_X = np.rot90(random_X, k=k_rot)
			# random_Y = np.rot90(random_Y, k=k_rot)
			X.append(random_X*intensity)
			y.append(random_Y)
		except:
			pass
				
	return X, y


# List to Numpy
def turn_2d_patches_list_to_numpy(x_imgs, y_imgs):
		
	x_imgs=np.asarray(x_imgs)
	x_imgs=x_imgs.astype(np.float32)
	
	x_imgs = np.reshape(x_imgs,(x_imgs.shape[0],x_imgs.shape[1],x_imgs.shape[2],1))

	y_imgs=np.asarray(y_imgs)
	y_imgs = one_hot_encode_mask(y_imgs)

	return x_imgs, y_imgs

# Crop scan to remove empty spaces
def crop_scan_mask(s, m, offset=0):
	x_start=0
	x_end = s.shape[0]
	y_start=0
	y_end = s.shape[1]
	z_start=0
	z_end = s.shape[2]

	for i in range(s.shape[0]):
		if np.max(s[i+offset,:,:])>=-1024:
			x_start = i
			break

	for i in reversed(range(s.shape[0])):
		if np.max(s[i-offset,:,:])>=-1024:
			x_end = i
			break

	for i in range(s.shape[1]):
		if np.max(s[:,i+offset,:])>=-1024:
			y_start = i
			break

	for i in reversed(range(s.shape[1])):
		if np.max(s[:,i-offset,:])>=-1024:
			y_end = i
			break

	for i in range(s.shape[2]):
		if np.max(s[:,:,i+offset])>=-1024:
			z_start = i
			break

	for i in reversed(range(s.shape[2])):
		if np.max(s[:,:,i-offset])>=-1024:
			z_end = i
			break

	return(x_start, x_end, y_start, y_end, z_start, z_end)

def zoom_mask(m, offset=0):
	
	'''
	mask needs be a 2-dim numpy array
	offset needs to be a positive integer >= 0
	offset provides some padding for the cropped mask area
	'''
	x_start = 0
	x_end = m.shape[0]
	y_start = 0
	y_end = m.shape[1]
	
	for i in range(m.shape[0]):
		if np.max(m[i+offset,:])>0:
			x_start = i
			break
	
	for i in reversed(range(m.shape[0])):
		if np.max(m[i-offset,:])>0:
			x_end = i
			break

	for i in range(m.shape[1]):
		if np.max(m[:,i+offset])>0:
			y_start = i
			break
	
	for i in reversed(range(m.shape[1])):
		if np.max(m[:,i-offset])>0:
			y_end = i
			break

	return m[x_start:x_end, y_start:y_end]

