# NVIDIA Architecture
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D
from keras.layers.core import Dropout
from keras import regularizers


# @return keras model
def model(weights=False):

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Conv2D(filters=24, kernel_size=(5, 5),
                     strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=(5, 5),
                     strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=(5, 5),
                     strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128,kernel_regularizer=regularizers.l2(0.07)))
    model.add(Dense(60))
    model.add(Dropout(.3))
    model.add(Dense(1))

    if weights:
        model.load_weights('NVIDIA.h5')

    return model