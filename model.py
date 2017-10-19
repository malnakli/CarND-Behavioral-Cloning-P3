import pandas as pd
import cv2
import numpy as np
from network_models import Basic, LeNet, NVIDIA

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
        for image in headers[:3]:
            row[image] = './' + '/'.join(row[image].split('/')[-3:])

        return row

    driving_log = driving_log.apply(get_image_path, axis=1)

    images = []
    steerings = []

    def get_training_data(row):
        # create adjusted steering measurements for the side camera images
        for header, correction in zip(headers[:3], [0, 0.1, -0.1]):
            image = cv2.imread(row[header])
            steering = float(row['steering'])
            images.append(image)
            steerings.append(steering + correction)

            # For helping with the left turn bias involves flipping images and taking the opposite sign of the steering measurement.
            image_flipped = np.fliplr(image)
            images.append(image_flipped)
            steerings.append(-(steering + correction))

    driving_log.apply(get_training_data, axis=1)

    # convert to numpy array since this what keras required
    x_train = np.array(images)
    y_train = np.array(steerings)

    return x_train, y_train


# ===================================================================== #
# ======================== Build Network Model ======================== #
# ===================================================================== #


def run_model(x, y, netModel='Basic'):
    # look at network model for more details for inside the model architecture

    models = {
        'Basic': Basic.model(),
        'LeNet': LeNet.model(),
        'NVIDIA': NVIDIA.model()
    }
    model = models[netModel]
    # TODO what are the differences among loss function and optimizer as well?
    # http://ruder.io/optimizing-gradient-descent/

    model.compile(loss='mse', optimizer='adam')
    model.fit(x, y, batch_size=32, nb_epoch=2,
              shuffle=True, validation_split=0.2)
    save_file_name = str(netModel) + '.h5'
    model.save(save_file_name)


# ===================================================================== #
# ======================== Execute code ======================== #
# ===================================================================== #

x_train, y_train = load_training_data()
print("x_train.shape=", x_train.shape)
print("y_train.shape=", y_train.shape)
print('data loaded')

run_model(x_train, y_train, netModel='NVIDIA')
