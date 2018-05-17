from step1_resize import *
from step2_random_take_data import *
from step3_shuffle_and_label import *

image_width = 227
image_height = 227

folder_include_all_classes = 'C:/data/HogwartsHouses/raw_data'
# for folder in os.listdir(mother_folder):
resize_to_somesize(folder_include_all_classes, image_width, image_height)

folder_of_resized_picture = 'C:/data/HogwartsHouses/Final_data%dby%d' % (image_width, image_height)
save_path = 'C:/data/HogwartsHouses/dataset_%dby%d' % (image_width, image_height)

random_take_data(folder_of_resized_picture, 0.8, save_path)

data_set_folder = 'C:/data/HogwartsHouses/dataset_%dby%d' % (image_width, image_height)
delete_old_pickle(data_set_folder + '/train_data')
delete_old_pickle(data_set_folder + '/test_data')
shuffle_and_label(data_set_folder + '/train_data', image_width, image_height)
shuffle_and_label(data_set_folder + '/test_data', image_width, image_height)