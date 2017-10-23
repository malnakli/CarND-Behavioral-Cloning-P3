from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Reshape, ZeroPadding2D, Input


def inception():
    inputs = Input(shape=(160, 320, 3))
    x = Lambda(lambda x: (x / 255.0) - 0.5)(inputs)
    x = Cropping2D(cropping=((65, 20), (0, 0)))(x)
    # InceptionV3 need to be at least (150, 150, 3)
    x = ZeroPadding2D(padding=((60, 15), (0, 0)))(x)

    x = InceptionV3(include_top=False, weights='imagenet', input_tensor=x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def vgg(layer='16'):
    inputs = Input(shape=(160, 320, 3))
    x = Lambda(lambda x: (x / 255.0) - 0.5)(inputs)
    x = Cropping2D(cropping=((65, 20), (0, 0)))(x)
    # vgg need to be at least (200, 200, 3)
    x = ZeroPadding2D(padding=((85, 40), (0, 0)))(x)
    if layer == '16':
        x = VGG16(
            include_top=False, weights='imagenet', input_tensor=x)
    elif layer == '19':
        x = VGG19(
            include_top=False, weights='imagenet', input_tensor=x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
