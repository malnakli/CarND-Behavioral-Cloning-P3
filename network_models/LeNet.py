from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout


# @return keras model
def model(weights=False):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(85, 300, 3)))
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dropout(.5))
    model.add(Dense(1))

    if weights:
        model.load_weights('LeNet.h5')

    return model
