from keras.applications.mobilenet import MobileNet
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Input
from keras.layers.core import Dropout
from keras import regularizers


def model(weights=True,freeze_pertrain_layers=True):
    inputs = Input(shape=(224, 224, 3), name='mobilenet_input')
    op = Lambda(lambda x: (x / 255.0) - 0.5)(inputs)
    # input must have a static square shape (one of (128,128), (160,160), (192,192), or (224, 224))
    app_model = MobileNet(include_top=False,
                        weights='imagenet',input_shape=(224,224,3), input_tensor=op,dropout=1e-3)

    # I freeze training all the convolutional layers, for two main reasons
        #
        # 1. my samples data is  small ~ 20k, so re-train first few convolutional layers, 
        #   it does not help improving the accurse of my output since most 
        #   the features is smiler (edge detection, shapes etc.)
        #
        # 2. When I try to retrain each convolutional layers, I ran out of memory when run on 4GB GPU.
        # 3. Very slow to train in 8GB GPU, each epoch it takes 5-8 minutes.
        
    for layer in app_model.layers:
        layer.trainable = not freeze_pertrain_layers # trainable has to be false in order to freez the layers

    op = Flatten(input_shape=app_model.output_shape[1:])(app_model.output)
    op = Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu')(op)
    op = Dense(64, activation='relu')(op)
    op = Dropout(.25)(op)
    outputs =  Dense(1)(op)

    model =  Model(inputs=inputs, outputs=outputs)

    if weights:
        model.load_weights('MobileNet.h5')

    return model
