import numpy as np
import os
import os.path as path
from six.moves import cPickle as pickle
import cv2

# image size, must be same as the values defined in 'step1_resize'
image_width = 100
image_height = 100


def one_hot(y_value, num_class):
    # delicate index operation, y_value is a 1D array, num_class is the amount of class (i.e for MNIST, num_class = 10)
    # reference: https://stackoverflow.com/questions/29831489/numpy-1-hot-array
    hot = np.zeros((len(y_value), num_class))
    hot[np.arange(len(y_value)), y_value] = 1
    return hot


# shuffle + label -> transfer into pickle file, folder is the train or test data folder
def shuffle_and_label(folder):
    image_files = os.listdir(folder)
    np.random.shuffle(image_files)
    data_set = np.ndarray(shape=(len(image_files), image_width, image_height, 3), dtype=np.float32)
    label_set = np.ndarray(shape=(len(image_files)), dtype=np.int)
    image_amount = 0
    for image in image_files:
        # strange part, when loading into 4d array, it automatically scales up, so need to divide 255
        data_set[image_amount] = cv2.imread(path.join(folder, image)) / 255
        if image[0] == 'R':
            label_set[image_amount] = 0
        elif image[0] == 'G':
            label_set[image_amount] = 1
        elif image[0] == 'S':
            label_set[image_amount] = 2
        elif image[0] == 'H':
            label_set[image_amount] = 3
        image_amount = image_amount + 1
    label_set = one_hot(label_set, 4)
    try:
        with open(path.join(folder, '00_data.pickle'), 'wb') as f:
            pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)
        with open(path.join(folder, '00_label.pickle'), 'wb') as f:
            pickle.dump(label_set, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', path.join(folder, 'XX.pickle'), 'wb', ':', e)


shuffle_and_label('./train_data')
shuffle_and_label('./test_data')