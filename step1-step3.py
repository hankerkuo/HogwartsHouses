from step1_resize import *
from step2_random_take_data import *
from step3_shuffle_and_label import *

# custom field, save_path is the final place to put 'train_data' and 'test_data' folders
image_width = 1024
image_height = 768
Momfolder_of_raw_data = 'C:/data/HogwartsHouses'
save_path = 'C:/data/HogwartsHouses/dataset_%dby%d' % (image_width, image_height)

# data path (DO NOT modify them)
f_raw_data = Momfolder_of_raw_data + '/raw_data'
f_resized_picture = Momfolder_of_raw_data + '/Final_data%dby%d' % (image_width, image_height)
data_set_folder = Momfolder_of_raw_data + '/dataset_%dby%d' % (image_width, image_height)

# resize -> take data from each class -> shuffle (no need to modify)
resize_to_somesize(f_raw_data, image_width, image_height)
random_take_data(f_resized_picture, 0.8, save_path)
delete_old_pickle(data_set_folder + '/train_data')
delete_old_pickle(data_set_folder + '/test_data')
shuffle_and_label(data_set_folder + '/train_data', image_width, image_height)
shuffle_and_label(data_set_folder + '/test_data', image_width, image_height)