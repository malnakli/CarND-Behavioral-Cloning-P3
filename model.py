import tensorflow as tf
from keras.callbacks import TensorBoard
import os
from helpers.prepare_data import split_data,combine_data,get_dir_names
from helpers.load_data import load_data_generator,load_data
from helpers.helper_functions import data_info
# NOTE: I am using keras 2.0.8 with tensorflow 1.3.0

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
    callbacks = None
    # to run tensoerborad: tensorboard --logdir=tb_logs
    if FLAGS.tb > 0:
        # example: tb_logs/NIVDIA/run2
        log_dir = './tb_logs/' + netModel + '/run' + str(FLAGS.tb) 
        tbCallback = TensorBoard(log_dir=log_dir, histogram_freq=1,
                                write_graph=FLAGS.tb_graph, write_images=True)
        callbacks=[tbCallback]                   


    # https://github.com/fchollet/keras/pull/8023 bug in shuffle argument in keras 2.0.8
    model.fit_generator(**fit_kwargs, callbacks=callbacks)

    save_file_name = str(netModel) + '.h5'
    model.save(save_file_name)


# ===================================================================== #
# ======================== Execute code ======================== #
# ===================================================================== #
def main():
    csv_file_path = combine_data(get_dir_names(),rec_data=FLAGS.rec_data)
    train, valid = split_data(csv_file_path)
    batch_size = FLAGS.bs

    train_generator = load_data_generator(train,FLAGS, batch_size)
    valid_generator = load_data_generator(valid,FLAGS, batch_size)

    model_fit_generator_arguments = {
        'generator': train_generator,
        'steps_per_epoch': int(train.shape[0] / batch_size),
        'validation_data': valid_generator,
        'validation_steps': int(valid.shape[0] / batch_size),
        'verbose': 1,
        'epochs': FLAGS.ep
    }

    if FLAGS.tb > 0:
        # if you would like to use TensorBorad histograms,
        # then do not used a generator for validation
        # look at this issue https://github.com/fchollet/keras/issues/3358
        model_fit_generator_arguments['validation_data'] = load_data(valid,FLAGS)
        # validation_steps Only relevant if validation_data is a generator
        model_fit_generator_arguments['validation_steps'] = None

    data_info(train, valid,FLAGS)
    run_model(model_fit_generator_arguments, netModel=FLAGS.model)


if __name__ == "__main__":
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('ep', 5, "epochs")
    flags.DEFINE_string(
        'model', 'NVIDIA', "one of: Basic, NVIDIA, LeNet, inception, vgg16, vgg19")
    flags.DEFINE_integer('bs', 32, "batch size for generator")
    flags.DEFINE_string('tf_debug', '3', "tensorflow debug mode: 0, 1, 2, 3")
    flags.DEFINE_integer('tb', 1, "TensorBoard, disable if <= 0, the number of run ")
    flags.DEFINE_boolean('tb_graph', True, "Print the graph of neural network as defined internally ")
    flags.DEFINE_integer(
        'img_use', 1, "1 to use center image, 2 to use both center and left image , 3 for all")
    flags.DEFINE_boolean(
        'flip_img', True, "generate more data by flip images and negate steering angle")
    flags.DEFINE_boolean('rec_data', True,
                         "rerun combine_data function to combine all the data")
    flags.DEFINE_boolean('plw', False, "pre load weight")
    # example:
    # python model.py --img_use 1 --ep 10 --model NVIDIA  --bs 32 --tf_debug 3 --tb

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = FLAGS.tf_debug
    main()
