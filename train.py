!pip install pynrrd
import nrrd
import matplotlib.pyplot as plt
import keras
import random
import gc
import numpy as np
import datetime

#%tensorflow_version 1.x
import tensorflow
print(tensorflow.__version__)

import tensorflow.keras

!python --version

print(tensorflow.__version__)
print(keras.__version__)

# Load the TensorBoard notebook extension
%load_ext tensorboard


import os
print(os.getcwd())
os.chdir('/content/drive/MyDrive/01_eom_segmentation/02_training/20220605')
print(os.getcwd())

from helpers.data import *
from helpers.datagenerator import *
from helpers.model import *
from helpers.losses import *



data_dir = '/content/drive/MyDrive/01_eom_segmentation/01_data/for_review'
#os.listdir(data_dir)

scan_paths  = []

for dirpaths, dirnames, filenames in os.walk(data_dir):
    if 'eom_cor-label.nrrd' in filenames:
        scan_paths.append(dirpaths)

print(len(scan_paths))
scan_paths=sorted(scan_paths)
print(scan_paths)

# Get the unique patient IDs

scan_patient_ids_anon = []

for i in range(len(scan_paths)):
    #print("\n Scan: "+str(i))
    scan_struct = scan_paths[i].split('/')
    #print(scan_struct)
    #print(len(scan_struct))
    #print(scan_struct[len(scan_struct)-2])
    scan_patient_ids_anon.append(scan_struct[len(scan_struct)-2])

print(len(scan_patient_ids_anon))
patient_ids_anon = sorted(list(set(scan_patient_ids_anon)))
print(len(patient_ids_anon))
print(patient_ids_anon)
patient_ids_anon_shuf = patient_ids_anon
random.seed(0)
random.shuffle(patient_ids_anon_shuf)
print(patient_ids_anon_shuf)

train_patients=patient_ids_anon_shuf[0:train_length]
test_patients =patient_ids_anon_shuf[train_length:train_length+test_length]

print(train_patients)
print(test_patients)


train_scan_paths = []
test_scan_paths = []

for scan_path in scan_paths:
    splits = scan_path.split('/')
    patient_id = splits[len(splits)-2]
    #print(patient_id)

    if patient_id in train_patients:
        train_scan_paths.append(scan_path)

    elif patient_id in test_patients:
        test_scan_paths.append(scan_path)    

print(len(train_scan_paths))
print(len(test_scan_paths))


#patch_sz=64
patch_sz=128

num_scans_in_batch = 4
num_patches_in_batch = 100
num_epochs = 50

unet_depth = 4

# Set loss function

# loss_function="wbce"                                # "dice" or "focal" - use focal tversky loss; dice loss is non-convex. 
weighted_binary_crossentropy = create_weighted_binary_crossentropy(zero_weight=0.1, one_weight=0.9)
#training_loss=weighted_binary_crossentropy          

# loss_function="dice"                               # "dice" or "focal" - use focal tversky loss; dice loss is non-convex. 
# training_loss=dice_coef_loss                       # "dice_coef_loss" or "focal_tversky"

loss_function="dice_wbce_loss"                      # "dice" or "focal" - use focal tversky loss; dice loss is non-convex. 
training_loss=dice_wbce_loss                        # "dice_coef_loss" or "focal_tversky"

#loss_function="focal"
#training_loss=focal_tversky


unet=unet_2d_model_deep4(n_classes=8, im_sz=patch_sz, n_channels=1, n_filters_start=64, growth_factor=2, upconv=True)
unet.summary()

def run_cv_training(train_scan_paths_cv, validation_scan_paths_cv):

    unet=unet_2d_model_deep4(n_classes=8, im_sz=patch_sz, n_channels=1, n_filters_start=64, growth_factor=2, upconv=True)
    #unet.summary()

    opt = Adam(learning_rate = 1e-4)
    unet.compile(optimizer=opt, 
                 loss=gl_sl_wrapper(alpha), 
                 metrics=[dice_coef_loss, mean_iou])

    # Set iteration string
    # The iteration with specified hyperparameters will be saved using this suffix
    iteration_suffix=str("unet2d"+
                        "_"+str(unet_depth)+"deep"+ 
                        "_"+str(loss_function)+"ls"+
                        "_"+str(cv)+"cv"
                        )

    results_dir=os.path.join('./', iteration_suffix)
    log_dir = str(results_dir)+'/'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(results_dir): os.mkdir(results_dir)
    if not os.path.exists(log_dir): os.mkdir(log_dir)

    print("Iteration suffix is: "+str(iteration_suffix))
    print("Results directory: "+str(results_dir))
    print("Log directory for Tensorboard: "+str(log_dir))


    training_params = {'scan_paths':train_scan_paths_cv, 
                    'num_scans_in_batch':num_scans_in_batch, 
                    'num_patches_in_batch' : num_patches_in_batch, 
                    'dim':patch_sz,
                    'shuffle':True
                    }

    validation_params = {'scan_paths':validation_scan_paths_cv, 
                    'num_scans_in_batch':num_scans_in_batch, 
                    'num_patches_in_batch' : num_patches_in_batch, 
                    'dim':patch_sz,
                    'shuffle':True
                    }

    training_generator = DataGenerator(**training_params)
    validation_generator = DataGenerator(**validation_params)

    # Continue training for specific number of epochs
    csv_logger = CSVLogger(os.path.join(log_dir, "01_training_history"+str(iteration_suffix)+".csv"), append=True)
    weight_saver = ModelCheckpoint(os.path.join(log_dir, '02_unet'+str(iteration_suffix)+'.h5'), 
                                monitor='val_dice_coef_loss', 
                                save_best_only=True)

    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    #es = EarlyStopping(monitor='val_dice_coef_loss', mode='min', verbose=1, patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.000001)

    # TRAIN
    history =  unet.fit(x=training_generator, 
                        validation_data=validation_generator, 
                        epochs=100, #epochs=num_epochs, 
                        verbose=1, 
                        callbacks=[weight_saver, tensorboard_callback, reduce_lr, AlphaScheduler(alpha, update_alpha)])


import numpy as np

cv_iters = [1]

for cv in cv_iters:

    print("\n\n\n\n"+str(cv))

    val_start = (cv-1)*17
    val_end   = cv * 17

    train1_start = 0
    train2_end   = 178

    train_scan_paths_cv = train_scan_paths[train1_start:val_start] + train_scan_paths[val_end:train2_end]
    validation_scan_paths_cv = train_scan_paths[val_start:val_end]

    run_cv_training(train_scan_paths_cv, validation_scan_paths_cv)