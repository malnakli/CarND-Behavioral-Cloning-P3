from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Input
from keras.layers.core import Dropout


# Still have an issue run this on GPU with 4GB memroy
def model(weights=False):
    inputs = Input(shape=(224, 224, 3), name='InceptionV3_input')
    op = Lambda(lambda x: (x / 255.0) - 0.5)(inputs)

    if weights:
        app_model = InceptionV3(
            include_top=False, weights='imagenet', input_tensor=op)
    else:
        app_model = InceptionV3(
            include_top=False, weights=None, input_tensor=op)
    
    op = Flatten(input_shape=app_model.output_shape[1:])(app_model.output)
    op = Dense(128, activation='relu')(op)
    op = Dense(64, activation='relu')(op)
    op = Dropout(.5)(op)
    outputs =  Dense(1)(op)

    return Model(inputs=inputs, outputs=outputs)
