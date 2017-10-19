from keras.models import Sequential
from keras.layers import Flatten, Dense


# @return keras model
def model():
    model = Sequential()
    model.add(Flatten(input_shape=(160, 320, 3)))
    model.add(Dense(1))

    return model
