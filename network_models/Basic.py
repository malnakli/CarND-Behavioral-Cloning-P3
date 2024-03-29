from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
import tensorflow as tf


# @return keras model
def model(weights=False):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(32, 32, 3)))
    model.add(Flatten())
    # it is used for tensorBorad
    with tf.name_scope('connect_layers'):
        model.add(Dense(128))
        model.add(Dense(64))
        model.add(Dense(1))

    if weights:
        model.load_weights('Basic.h5')

    return model
