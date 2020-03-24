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
    train_file_path = '..//fingers//fingers//train//'
else:
    train_file_path = sys.argv[1]

print (train_file_path)

img_data, label_hand_data, label_count_data = [], [], []

for file in glob.glob(file_path):
    label = file.split("_")[1].split(".")[0]
    label_hand = list(label)[0]
    label_count = list(label)[1]

    img = load_img(
        img_path, 
        target_size = (128, 128)
    )

    x = img_to_array(img)
    label_hand_data.append(label_hand)
    label_hand_count.append(label_count)

