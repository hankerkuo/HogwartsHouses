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
def shuffle_and_label(folder, image_width, image_height):
    image_files = os.listdir(folder)
    np.random.shuffle(image_files)
    data_set = np.ndarray(shape=(len(image_files), image_width, image_height, 3), dtype=np.float32)
    label_set = np.ndarray(shape=(len(image_files)), dtype=np.int)
    image_amount = 0

    for image in image_files:
        # strange part, when loading into 4d array, it automatically scales up, so need to divide 255
        data_set[image_amount] = cv2.imread(path.join(folder, image)) / 255
        # label the four houses in Hogwarts (R -> Ravenclaw, G -> Gryffindor, S -> Slytherin, H -> Hufflepuff)
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
            print('successfully made pickle:', path.join(folder, '00_data.pickle'))
        with open(path.join(folder, '00_label.pickle'), 'wb') as f:
            pickle.dump(label_set, f, pickle.HIGHEST_PROTOCOL)
            print('successfully made pickle:', path.join(folder, '00_label.pickle'))
    except Exception as e:
        print('Unable to save data to', path.join(folder, 'XX.pickle'), 'wb', ':', e)


# shuffle + label -> transfer into pickle file, folder is the train or test data folder
# gray-scale version
def shuffle_and_label_grayscale(folder, image_width, image_height):
    image_files = os.listdir(folder)
    np.random.shuffle(image_files)
    data_set = np.ndarray(shape=(len(image_files), image_width, image_height), dtype=np.float32)
    label_set = np.ndarray(shape=(len(image_files)), dtype=np.int)
    image_amount = 0
    for image in image_files:
        # strange part, when loading into 4d array, it automatically scales up, so need to divide 255
        # gray-scale version
        data_set[image_amount] = cv2.imread(path.join(folder, image), cv2.IMREAD_GRAYSCALE) / 255
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
            print('successfully made pickle:', path.join(folder, '00_data.pickle'))
        with open(path.join(folder, '00_label.pickle'), 'wb') as f:
            pickle.dump(label_set, f, pickle.HIGHEST_PROTOCOL)
            print('successfully made pickle:', path.join(folder, '00_label.pickle'))
    except Exception as e:
        print('Unable to save data to', path.join(folder, 'XX.pickle'), 'wb', ':', e)


# used to delete the existing pickle files
def delete_old_pickle(folder):
    try:
        os.remove(path.join(folder, '00_data.pickle'))
        print('successfully remove old pickle:', path.join(folder, '00_data.pickle'))
    except Exception as e:
        print('Unable to delete file', path.join(folder, '00_data.pickle'), 'error : ', e)
    try:
        os.remove(path.join(folder, '00_label.pickle'))
        print('successfully remove old pickle:', path.join(folder, '00_label.pickle'))
    except Exception as e:
        print('Unable to delete file', path.join(folder, '00_label.pickle'), 'error : ', e)

# sample code of using this lib
'''
data_set_folder = 'C:/data/HogwartsHouses/dataset_32by32'
delete_old_pickle(data_set_folder + '/train_data')
delete_old_pickle(data_set_folder + '/test_data')
shuffle_and_label(data_set_folder + '/train_data', 32, 32)
shuffle_and_label(data_set_folder + '/test_data', 32, 32)
'''