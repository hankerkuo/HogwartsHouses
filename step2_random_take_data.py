import os
import os.path as path
import numpy as np
import shutil


# randomly take training data and testing data, and then put them in the folder of this project!
# mother_folder must contain all of the class folders
# train_ratio argument is the ratio of training data, its value must between [0, 1]
def random_take_data(mother_folder, train_ratio, save_path):
    # create folders
    if not path.exists(path.join(save_path, 'train_data')):
        os.makedirs(path.join(save_path, 'train_data'))
        print('successfully creat folder : train_data at', path.join(save_path, 'train_data'))
    if not path.exists(path.join(save_path, 'test_data')):
        os.makedirs(path.join(save_path, 'test_data'))
        print('successfully creat folder : test_data at', path.join(save_path, 'test_data'))

    print('Processing data ...')
    for kid_folder in os.listdir(mother_folder):                                        # go through each class
        image_files = os.listdir(path.join(mother_folder, kid_folder))                  # read all the files in a kid folder (e.g. each class)
        train_num = np.floor(len(image_files) * train_ratio)
        image_files_train = np.random.choice(image_files, np.int32(train_num), replace=False)     # randomly choosing train data
        image_files_test = np.setdiff1d(image_files, image_files_train)                 # the remaining data becomes test data
        for image in image_files_train:                                                 # copy the images to new folders
            shutil.copy(path.join(mother_folder, kid_folder, image), path.join(save_path, 'train_data'))
        for image in image_files_test:
            shutil.copy(path.join(mother_folder, kid_folder, image), path.join(save_path, 'test_data'))
    print('Process done')

# sample code of using this lib
'''
folder_of_resized_picture = 'C:/data/HogwartsHouses/Final_data32by32'
save_path = 'C:/data/HogwartsHouses/dataset_32by32'
random_take_data(folder_of_resized_picture, 0.8, save_path)
'''