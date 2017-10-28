from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Input
from keras.layers.core import Dropout


# Still have an issue run this on GPU with 4GB memroy
def model(weights=True,freez_pertrian_layers=True):
    inputs = Input(shape=(200, 200, 3), name='vgg_input')
    op = Lambda(lambda x: (x / 255.0) - 0.5)(inputs)

    if weights:
        app_model = VGG16(include_top=False,
                          weights='imagenet', input_tensor=op)
        # I freez train all the convolutional layers, for two main reasons
        # 1. my samples data is  small ~ 20k
        #   a. so retrain frist few convolutiional layers, 
        #       it does not help improve the accurse of my output since most 
        #       the features is smiler (edge detection, shapes etc.)
        #   b. from training the last two convolutiional layers, it does (not) help.
        # 2. When I try to retrain each convolutional layers, I ran out of memory when run on 4GB GPU.
        for layer in app_model.layers:
            layer.trainable = not freez_pertrian_layers

    else:
        # Is not recommend to use this for small data.
        app_model = VGG16(include_top=False, weights=None, input_tensor=op)

    op = Flatten(input_shape=app_model.output_shape[1:])(app_model.output)
    op = Dense(128, activation='relu')(op)
    op = Dense(64, activation='relu')(op)
    op = Dropout(.5)(op)
    outputs =  Dense(1)(op)

    return Model(inputs=inputs, outputs=outputs)
