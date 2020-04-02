#
# Created on Tue Mar 24 2020 by Viramya
#
# Copyright (c) 2020 Your Company
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import glob, os, sys

from keras_preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.models import Sequential, Input
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')


if len(sys.argv) < 2:
    train_file_path = "D://Data//Counting Finger//fingers//train"
else:
    train_file_path = sys.argv[1]

file_path = "{}//*.png".format(train_file_path, )

img_data, label_hand_data, label_count_data = [], [], []

for file in glob.glob(file_path):
    label = file.split("_")[1].split(".")[0]
    label_count = list(label)[0]
    label_hand = list(label)[1]

    img = load_img(
        file, 
        target_size = (128, 128)
    )

    img_data.append(img_to_array(img))
    label_hand_data.append(label_hand)
    label_count_data.append(label_count)

img_data = np.array(img_data)
hand_data = np.array(label_hand_data)
count_data = np.array(label_count_data)

print ("DATA LOADED")

'''
    Defining the ImageDataGenerator class from Keras
'''

datagen = ImageDataGenerator(
    featurewise_center = False, 
    samplewise_center = False, 
    featurewise_std_normalization = False, 
    samplewise_std_normalization = False, 
    zca_whitening = False, 
    zca_epsilon = 1e-06, 
    rotation_range = 0, 
    width_shift_range = 0.0, 
    height_shift_range = 0.0, 
    brightness_range = None, 
    shear_range = 0.0, 
    zoom_range = 0.0, 
    channel_shift_range = 0.0, 
    fill_mode = 'nearest', 
    cval = 0.0, 
    horizontal_flip = False, 
    vertical_flip = False, 
    rescale = None, 
    preprocessing_function = None, 
    data_format = 'channels_last', 
    validation_split = 0.0, 
    interpolation_order = 1, 
    dtype='float32'
)