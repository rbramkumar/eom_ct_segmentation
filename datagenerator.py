from tensorflow.keras.utils import Sequence
from helpers.data import *
import nrrd
import numpy as np
import os
import gc
import random



class DataGenerator(Sequence):
	
	'Generates data for Keras'
	'IMPORTANT NOTE: SCAN IDs go from 1-total inclusive but indexes go from 0-(total-1) since we use np.arange'
	def __init__(self, scan_paths, num_scans_in_batch, num_patches_in_batch, dim, shuffle):
		
		'Initialization'
		self.scan_paths = scan_paths
		self.num_scans_in_batch = num_scans_in_batch
		self.num_patches_in_batch = num_patches_in_batch
		self.dim = dim
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		'In this case we move 1 scan after another - pick num_scans_in_batch and generate patches'
		return int(len(self.scan_paths)-self.num_scans_in_batch+1)

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index : (index+self.num_scans_in_batch)]
		#print("Indexes")
		#print(indexes)

		# Find list of IDs
		scan_paths_temp = [self.scan_paths[k] for k in indexes]
		#print("scan_ID_temp")
		#print(scan_IDs_temp)

		# Generate data
		#print("\nGenerating data for batch: "+str(scan_IDs_temp))
		X, y = self.__data_generation(scan_paths_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.scan_paths))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, scan_paths_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

		scans = []
		masks = []
		
		for scan_path in scan_paths_temp:

			dirpaths = scan_path
			filenames = os.listdir(dirpaths)

			nrrds = [s for s in filenames if "nrrd" in s]
			scan_nrrd = [s for s in nrrds if "COR" in s]
			mask_nrrd = 'eom_cor-label.nrrd'
			
			s, h = nrrd.read(os.path.join(dirpaths, scan_nrrd[0]))
			m, h = nrrd.read(os.path.join(dirpaths, 'eom_cor-label.nrrd'))

			s[s<-1024]=-1025        
			x1,x2,y1,y2,z1,z2 = crop_scan_mask(s,m,offset=25)
			s = s[x1:x2,y1:y2,z1:z2]
			m = m[x1:x2,y1:y2,z1:z2]
			
			s = window_image(image=s, window_center=50, window_width=250)
			scans.append(s)
			masks.append(m)
		
		scans, masks = get_2d_cor_patches(scans, masks, num_patches_required = self.num_patches_in_batch, sz = self.dim, scale_factor=(80,120), intensity_factor=(90,110))

		scans, masks = turn_2d_patches_list_to_numpy(scans, masks)

		X=scans.astype(np.float32)
		y=masks.astype(np.float32)
		
		# Conservatively release memory to delete the variables we don't need anymore
		# The scans will be written over in the next step (batch) anyway
		del scans, masks
		gc.collect()
		
		return X, y