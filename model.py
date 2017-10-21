import pandas as pd
import cv2
import numpy as np
from network_models import Basic, LeNet, NVIDIA
from keras.callbacks import TensorBoard
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os

# NOTE: I am using keras 2.0.8

# ===================================================================== #
# ========================  Combine Data ======================== #
# ===================================================================== #

# NOTE: The reason I am using pandas for even thought is not necessrary,
# I would like to improve my skill on pandas.

def get_dir_names():
    folders = [] 
    root_dir, dirs, files = next(os.walk('./data'))
    for name in dirs:
        folders.append(os.path.join(root_dir, name))
    return  folders

# create_all the driving logs into new one
# @return path of driving log file
def combine_data(folders):
    list_   = []
    headers = ['center_image', 'left_image', 'right_image',
               'steering', 'throttle', 'break', 'speed']
    for folder in folders:
        df = pd.read_csv(folder + "/driving_log.csv",header=None,names=headers)
        # get the relative path of the images
        def get_image_path(row):
            for image in headers[:3]:
                row[image] = folder + '/' + '/'.join(row[image].split('/')[-2:])

            return row

        df = df.apply(get_image_path, axis=1)
        list_.append(df)
    
    driving_log_file = "./data/combine_driving_log.csv"
    pd.concat(list_).to_csv(driving_log_file,index=False)
   
    return driving_log_file

# ===================================================================== #
# ========================  split Data ======================== #
# ===================================================================== #

# @return pandas DataFrame
def split_data(path_to_csv):
    # read  driveing log csv file
    data = pd.read_csv(path_to_csv)
    return train_test_split(data, test_size=0.2)

# ===================================================================== #
# ========================  Load Data ======================== #
# ===================================================================== #

def load_data(samples):
    images = []
    steerings = []
    for index, row in samples.iterrows():        
        # create adjusted steering measurements for the side camera images
        for header, correction in zip(samples.columns[:3], [0, 0.1, -0.1]):
            image = cv2.imread(row[header])
            steering = float(row['steering'])
            images.append(image)
            steerings.append(steering + correction)

            # For helping with the left turn bias involves flipping images and taking the opposite sign of the steering measurement.
            # image_flipped = np.fliplr(image)
            # images.append(image_flipped)
            # steerings.append(-(steering + correction))

    # convert to numpy array since this what keras required
    return shuffle( np.array(images), np.array(steerings))

# @samples: panads dataFrame
def load_data_generator(samples,batch_size=32):
    num_samples = samples.shape[0]
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            yield load_data(batch_samples)



# ===================================================================== #
# ======================== Build Network Model ======================== #
# ===================================================================== #


def run_model(fit_kwargs,netModel='Basic' ):
    # look at network model for more details for inside the model architecture

    models = {
        'Basic': Basic.model(),
        'LeNet': LeNet.model(),
        'NVIDIA': NVIDIA.model()
    }
    model = models[netModel]
    # TODO what are the differences among loss function and optimizer as well?
    # http://ruder.io/optimizing-gradient-descent/

    model.compile(loss='mse', optimizer='adam')

    # NOTE: the following is a reference for me, 
    # Taken from: https://stackoverflow.com/questions/45265436/keras-save-image-embedding-of-the-mnist-data-set
    # embedding_layer_names = set(layer.name
    #                         for layer in model.layers
    #                         if layer.name.startswith('dense'))
    # embeddings_freq = 1

    #For tensorBoarb
    # write_graph: Print the graph of neural network as defined internally.
    # write_images: Create an image by combining the weight of neural network
    # histogram_freq: Plot the distributions of weights and biases in the neural network
    # embeddings_freq: frequency (in epochs) at which selected embedding layers will be saved.
    # embeddings_layer_names: a list of names of layers to keep eye on. If None or empty list all the embedding layer will be watched.

    tbCallback = TensorBoard(log_dir="./graph_logs", histogram_freq=1, 
    write_graph=True, write_images=True)

    # # https://github.com/fchollet/keras/pull/8023 bug in shuffle argument in keras 2.0.8
    model.fit_generator(**fit_kwargs , callbacks=[tbCallback])
    
    save_file_name = str(netModel) + '.h5'
    model.save(save_file_name)


# ===================================================================== #
# ======================== Execute code ======================== #
# ===================================================================== #
def main():
    train , valid = split_data(combine_data(get_dir_names()))
    batch_size = 32
    model_fit_generator_arguments = {
        'generator':load_data_generator(train,batch_size),
        'steps_per_epoch':int(train.shape[0]/batch_size), 
        # I would like to use TensorBorad histograms, 
        # look at this issue https://github.com/fchollet/keras/issues/3358
        'validation_data':load_data(valid),
    # 'validation_steps':int(valid.shape[0]/batch_size), # Only relevant if validation_data is a generator 
    'verbose':1,
    'epochs':1
    }
    run_model(model_fit_generator_arguments, netModel='Basic')

if __name__ == "__main__":
    main()

