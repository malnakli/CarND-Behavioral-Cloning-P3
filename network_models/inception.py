from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, ZeroPadding2D, Input

# Still have an issue run this on GPU with 4GB memroy


def model(weights=False):
    inputs = Input(shape=(85, 300, 3), name='InceptionV3_input')
    op = Lambda(lambda x: (x / 255.0) - 0.5)(inputs)
    # InceptionV3 need to be at least (150, 150, 3)
    op = ZeroPadding2D(padding=((60, 20), (0, 0)))(op)

    if weights:
        app_model = InceptionV3(
            include_top=False, weights='imagenet', input_tensor=op)
    else:
        app_model = InceptionV3(
            include_top=False, weights=None, input_tensor=op)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=app_model.output_shape[1:]))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dense(64, activation='relu'))
    top_model.add(Dense(1))

    model = Model(inputs=app_model.input, outputs=top_model(app_model.output))
    return model
