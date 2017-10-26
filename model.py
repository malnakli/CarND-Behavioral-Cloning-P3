import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os
# NOTE: I am using keras 2.0.8 with tensorflow 1.3.0


# ===================================================================== #
# ========================  Combine Data ======================== #
# ===================================================================== #

# NOTE: The reason that I am using pandas even thought is not necessrary,
# I would like to improve my skill on pandas.


def get_dir_names():
    """Obtain directory names where data are saved.
    Returns:
       List of folders names
    """
    folders = []
    root_dir, dirs, files = next(os.walk('./data'))
    for name in dirs:
        folders.append(os.path.join(root_dir, name))
    return folders


def combine_data(folders):
    """combine all driving logs files into one.
    Arguments:
        folders: List of folders names
    Returns:
       Path of the new driving log file.
    """
    driving_log_file = "./data/combine_driving_log.csv"
    if FLAGS.recombine_data:
        list_ = []
        headers = ['center', 'left', 'right',
                   'steering', 'throttle', 'break', 'speed']
        for folder in folders:
            df = pd.read_csv(folder + "/driving_log.csv",
                             header=None, names=headers)
            # get the relative path of the images

            def get_image_path(row):
                for image in headers[:3]:
                    row[image] = folder + '/' + \
                        '/'.join(row[image].split('/')[-2:])

                return row

            df = df.apply(get_image_path, axis=1)
            list_.append(df)

        pd.concat(list_).to_csv(driving_log_file, index=False)

    return driving_log_file

# ===================================================================== #
# ============================ split Data ============================= #
# ===================================================================== #

# @return pandas DataFrame


def split_data(path_to_csv):
    # read  driveing log csv file
    data = pd.read_csv(path_to_csv)
    return train_test_split(data, test_size=0.2)

# Calculate how much the length of data after augmentation


def len_multiplier():
    if FLAGS.flip_img:
        return FLAGS.img_use * 2
    else:
        return FLAGS.img_use

# ===================================================================== #
# ========================  Load Data ======================== #
# ===================================================================== #


def load_data(samples):
    images = []
    steerings = []
    for index, row in samples.iterrows():
        # create adjusted steering measurements for the side camera images
        corrections = [0, 0.3, -0.2]
        for header, correction in zip(samples.columns[:FLAGS.img_use], corrections[:FLAGS.img_use]):
            # image shape (160,320,3)
            image = cv2.imread(row[header], 0)
            # cut unusfull pixels from the image
            image = image[55:140, 10:310]  # (85,300)
            # reshape the image
            image = np.reshape(image, image.shape[:] + (1,))  # (85,300, 1)
            steering = float(row['steering'])
            images.append(image)
            steerings.append(steering + correction)
            if FLAGS.flip_img:
                # For helping with the left turn bias involves flipping images and taking the opposite sign of the steering measurement.
                image_flipped = cv2.flip(image, 1)
                image_flipped = np.reshape(
                    image_flipped, image_flipped.shape[:] + (1,))  # (85,300, 1)
                images.append(image_flipped)
                steerings.append(-(steering + correction))

    # convert to numpy array since this what keras required

    return shuffle(np.array(images), np.array(steerings))

# @samples: panads dataFrame


def load_data_generator(samples, batch_size=32):
    num_samples = samples.shape[0]
    while 1:  # Loop forever so the generator never terminates
        # The reason this should work is that over many epochs,
        # random selection should ensure that all of your training data is taken into account.
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]
            yield load_data(batch_samples)


# ===================================================================== #
# ======================== Build Network Model ======================== #
# ===================================================================== #


def run_model(fit_kwargs, netModel='NVIDIA'):
    # look at network model for more details for inside the model architecture
    # dynamically import modules
    import_package = "network_models." + netModel
    import_model = __import__(
        import_package, globals(), locals(), ['model'], 0)
    model = import_model.model(FLAGS.plw)

    # TODO what are the differences among loss function and optimizer as well?
    # http://ruder.io/optimizing-gradient-descent/

    model.compile(loss='mse', optimizer='adam')

    # NOTE: the following is a reference for me,
    # Taken from: https://stackoverflow.com/questions/45265436/keras-save-image-embedding-of-the-mnist-data-set
    # embedding_layer_names = set(layer.name
    #                         for layer in model.layers
    #                         if layer.name.startswith('dense'))
    # embeddings_freq = 1

    # For tensorBoarb
    # write_graph: Print the graph of neural network as defined internally.
    # write_images: Create an image by combining the weight of neural network
    # histogram_freq: Plot the distributions of weights and biases in the neural network
    # embeddings_freq: frequency (in epochs) at which selected embedding layers will be saved.
    # embeddings_layer_names: a list of names of layers to keep eye on. If None or empty list all the embedding layer will be watched.

    # to run tensoerborad: tensorboard --logdir=tb_logs
    if FLAGS.tb > 0:
        # example: tb_logs/NIVDIA/run2
        log_dir = './tb_logs/' + netModel + '/run' + str(FLAGS.tb) 
        tbCallback = TensorBoard(log_dir=log_dir, histogram_freq=1,
                                write_graph=True, write_images=True)
    else:
        tbCallback =None

    # # https://github.com/fchollet/keras/pull/8023 bug in shuffle argument in keras 2.0.8
    model.fit_generator(**fit_kwargs, callbacks=[tbCallback])

    save_file_name = str(netModel) + '.h5'
    model.save(save_file_name)


# ===================================================================== #
# ======================== Execute code ======================== #
# ===================================================================== #
def main():
    train, valid = split_data(combine_data(get_dir_names()))
    batch_size = FLAGS.bs
    model_fit_generator_arguments = {
        'generator': load_data_generator(train, batch_size),
        'steps_per_epoch': int(train.shape[0] / batch_size),
        'validation_data': load_data_generator(valid, batch_size),
        'validation_steps': int(valid.shape[0] / batch_size),
        'verbose': 1,
        'epochs': FLAGS.ep
    }

    if FLAGS.tb > 0:
        # if you would like to use TensorBorad histograms,
        # then do not used a generator for validation
        # look at this issue https://github.com/fchollet/keras/issues/3358
        model_fit_generator_arguments['validation_data'] = load_data(valid)
        # validation_steps Only relevant if validation_data is a generator
        model_fit_generator_arguments['validation_steps'] = None
    data_info(train, valid)
    run_model(model_fit_generator_arguments, netModel=FLAGS.model)


def data_info(train, valid):
    if int(FLAGS.tf_debug) <= 2:
        print("train data length: {:,} ".format(train.shape[0] * len_multiplier()))
        print("valid data length: {:,} ".format(valid.shape[0] * len_multiplier()))
        print("train data shape: ", train.shape)


if __name__ == "__main__":
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('ep', 5, "epochs")
    flags.DEFINE_string(
        'model', 'NVIDIA', "one of: Basic, NVIDIA, LeNet, inception, vgg16, vgg19")
    flags.DEFINE_integer('bs', 32, "batch size for generator")
    flags.DEFINE_string('tf_debug', '3', "tensorflow debug mode: 0, 1, 2, 3")
    flags.DEFINE_integer('tb', 1, "TensorBoard, disable if <= 0, the number of run ")
    flags.DEFINE_integer(
        'img_use', 1, "1 to use center image, 2 to use both center and left image , 3 for all")
    flags.DEFINE_boolean(
        'flip_img', True, "generate more data by flip images and negate steering angle")
    flags.DEFINE_boolean('recombine_data', True,
                         "rerun to combine all the data")
    flags.DEFINE_boolean('plw', False, "pre load weight")
    # example:
    # python model.py --img_use 1 --ep 10 --model NVIDIA  --bs 32 --tf_debug 3 --tb

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = FLAGS.tf_debug
    main()
