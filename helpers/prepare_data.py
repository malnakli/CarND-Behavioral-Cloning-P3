from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np

# ===================================================================== #
# ========================  Combine Data ======================== #
# ===================================================================== #

# NOTE: The reason that I am using pandas even thought is not necessrary,
# I would like to improve my skill on pandas.

def get_dir_names(FLAGS):
    """Obtain directory names where data are saved.
    Returns:
       List of folders names
    """
    folders = []
    root_dir, dirs, files = next(os.walk('./data'))
    for name in dirs:
        folders.append(os.path.join(root_dir, name))
    
    folders = np.array(folders)
    folders_include = FLAGS.folders_include.split(',')
    # add the root to the names of the folders
    folders_include = np.core.defchararray.add(root_dir + '/',folders_include)
    return  np.extract(np.isin(folders , folders_include),folders)


def combine_data(folders,rec_data=False):
    """combine all driving logs files into one.
    Arguments:
        folders: List of folders names
    Returns:
       Path of the new driving log file.
    """
    driving_log_file = "./data/combine_driving_log.csv"
    if rec_data:
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