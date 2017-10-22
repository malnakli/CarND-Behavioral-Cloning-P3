from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential,Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D,Reshape


def inception():
   # model = Sequential()
    #model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    #model.add(Cropping2D(cropping=((65, 20), (0, 0))))
    model = InceptionV3(include_top=False,weights='imagenet',input_shape=(160, 320, 3))
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(128,activation='relu'))
    top_model.add(Dense(64,activation='relu'))
    top_model.add(Dense(1))

    model = Model(inputs= model.input, outputs= top_model(model.output)) 

    return model