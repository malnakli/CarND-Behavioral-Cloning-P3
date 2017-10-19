import pandas as pd
import cv2
import numpy as np
from network_models import basic

# ===================================================================== #
# ========================  Load Training Data ======================== #
# ===================================================================== #


def load_training_data():

    # NOTE: The reason I am using pandas for even thought is not necessrary,
    # I would like to improve my skill on pandas.

    # read  driveing log csv file
    headers = ['center_image', 'left_image', 'right_image',
               'steering', 'throttle', 'break', 'speed']
    driving_log = pd.read_csv("./data/driving_log.csv",
                              header=None, names=headers)

    # get the relative path of the images

    def get_image_path(row):
        row['center_image'] = './' + \
            '/'.join(row['center_image'].split('/')[-3:])
        row['left_image'] = './' + '/'.join(row['left_image'].split('/')[-3:])
        row['right_image'] = './' + \
            '/'.join(row['right_image'].split('/')[-3:])
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

    return x_train, y_train


x_train, y_train = load_training_data()
print("x_train.shape=", x_train.shape)
print("y_train.shape=", y_train.shape)
print('data loaded')

# ===================================================================== #
# ======================== Build Network Model ======================== #
# ===================================================================== #


# look at network model for more details for inside the model architecture

model = basic.model()
# TODO what are the differences among loss function and optimizer as well?

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, batch_size=32, nb_epoch=25,
          shuffle=True, validation_split=0.2)

model.save('model.h5')
