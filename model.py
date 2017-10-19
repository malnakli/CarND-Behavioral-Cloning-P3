import pandas as pd
import cv2
import numpy as np

# ===================================================================== #
# ========================  Load Training Data ======================== #
# ===================================================================== #

# NOTE: The reason I am using pandas for even thought is not necessrary,
# I would like to improve my skill on pandas.

# read  driveing log csv file
headers = ['center_image', 'left_image', 'right_image',
           'steering', 'throttle', 'break', 'speed']
driving_log = pd.read_csv("./data/driving_log.csv", header=None, names=headers)

# get the relative path of the images


def get_image_path(row):
    row['center_image'] = './' + '/'.join(row['center_image'].split('/')[-3:])
    row['left_image'] = './' + '/'.join(row['left_image'].split('/')[-3:])
    row['right_image'] = './' + '/'.join(row['right_image'].split('/')[-3:])
    return row


driving_log = driving_log.apply(get_image_path, axis=1)

center_images = []
steerings = []


def get_training_data(row):
    center_images.append(cv2.imread(row['center_image']))
    steerings.append(row['steering'])


driving_log.apply(get_training_data, axis=1)
# convert to numpy array since this what keras required
x_train = np.array(center_images)
y_train = np.array(steerings)

print(x_train.shape)

# ===================================================================== #
# ======================== Build Network Model ======================== #
# ===================================================================== #
