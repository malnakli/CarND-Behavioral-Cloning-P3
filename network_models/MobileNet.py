from keras.applications.mobilenet import MobileNet
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Input
from keras.layers.core import Dropout
from keras import regularizers


def model(weights=True,freez_pertrian_layers=True):
    inputs = Input(shape=(224, 224, 3), name='mobilenet_input')
    op = Lambda(lambda x: (x / 255.0) - 0.5)(inputs)
    # input must have a static square shape (one of (128,128), (160,160), (192,192), or (224, 224))
    app_model = MobileNet(include_top=False,
                        weights='imagenet',input_shape=(224,224,3), input_tensor=op,dropout=1e-3)
    for layer in app_model.layers:
        layer.trainable = not freez_pertrian_layers # trainable has to be false in order to freez the layers

    op = Flatten(input_shape=app_model.output_shape[1:])(app_model.output)
    op = Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu')(op)
    op = Dense(64, activation='relu')(op)
    op = Dropout(.25)(op)
    outputs =  Dense(1)(op)

    model =  Model(inputs=inputs, outputs=outputs)

    if weights:
        model.load_weights('MobileNet.h5')

    return model
