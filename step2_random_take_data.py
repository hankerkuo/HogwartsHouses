import os
import os.path as path
import numpy as np
import shutil


# randomly take training data and testing data, and then put them in the folder of this project!
# mother_folder must contain all of the class folders
# train_ratio argument is the ratio of training data, its value must between [0, 1]
def random_take_data(mother_folder, train_ratio):
    # create folders
    if not path.exists('./train_data'):
        os.makedirs('./train_data')
    if not path.exists('./test_data'):
        os.makedirs('./test_data')

    for kid_folder in os.listdir(mother_folder):                                        # go through each class
        image_files = os.listdir(path.join(mother_folder, kid_folder))                  # read all the files in a kid folder (e.g. each class)
        train_num = np.floor(len(image_files) * train_ratio)
        image_files_train = np.random.choice(image_files, np.int32(train_num), replace=False)     # randomly choosing train data
        image_files_test = np.setdiff1d(image_files, image_files_train)                 # the remaining data becomes test data
        for image in image_files_train:                                                 # copy the images to new folders
            shutil.copy(path.join(mother_folder, kid_folder, image), './train_data')
        for image in image_files_test:
            shutil.copy(path.join(mother_folder, kid_folder, image), './test_data')


mother_folder = 'C:/data/HogwartsHouses/Final_data'
random_take_data(mother_folder, 0.8)