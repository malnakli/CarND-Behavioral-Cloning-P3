from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Reshape, ZeroPadding2D, Input


def inception():
    inputs = Input(shape=(160, 320, 3), name='InceptionV3_input')
    op = Lambda(lambda x: (x / 255.0) - 0.5)(inputs)
    op = Cropping2D(cropping=((65, 20), (0, 0)))(op)
    # InceptionV3 need to be at least (150, 150, 3)
    op = ZeroPadding2D(padding=((60, 20), (0, 0)))(op)

    app_model = InceptionV3(
        include_top=False, weights='imagenet', input_tensor=op)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=app_model.output_shape[1:]))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dense(64, activation='relu'))
    top_model.add(Dense(1))

    model = Model(inputs=app_model.input, outputs=top_model(app_model.output))
    return model


def vgg(layer=16):
    inputs = Input(shape=(160, 320, 3), name='vgg_input')
    op = Lambda(lambda x: (x / 255.0) - 0.5)(inputs)
    op = Cropping2D(cropping=((65, 20), (0, 0)))(op)
    # vgg need to be at least (200, 200, 3)
    op = ZeroPadding2D(padding=((85, 40), (0, 0)))(op)
    if layer == 19:
        app_model = VGG19(
            include_top=False, weights='imagenet', input_tensor=op)
    else:
        app_model = VGG16(
            include_top=False, weights='imagenet', input_tensor=op)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=app_model.output_shape[1:]))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dense(64, activation='relu'))
    top_model.add(Dense(1))

    model = Model(inputs=app_model.input, outputs=top_model(app_model.output))
    return model
