# Calculate how much the length of data after augmentation
def len_multiplier(FLAGS):
    if FLAGS.flip_img:
        return FLAGS.img_use * 2
    else:
        return FLAGS.img_use


def data_info(train, valid,FLAGS):
    if int(FLAGS.tf_debug) <= 2:
        print("train data length: {:,} ".format(train.shape[0] * len_multiplier(FLAGS)))
        print("valid data length: {:,} ".format(valid.shape[0] * len_multiplier(FLAGS)))
        print("train data shape: ", train.shape)

