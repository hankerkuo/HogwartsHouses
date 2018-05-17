import cv2
import os
import os.path as path


# This function resizes images (under the 'folder') to a custom size
def resize_to_somesize(folder, width, height):
    # mother_folder is the upper folder of folder
    mother_folder = path.abspath(folder + '/..')
    folder_final = path.join(mother_folder, 'Final_data%dby%d' % (width, height))

    # create 'Final_data' folder
    if not path.exists(folder_final):
        os.makedirs(folder_final)
        print('Successfully created', folder_final)

    # load all the classes folder name
    for classes_folder in os.listdir(folder):

        # folder_Final is the new folder named 'Final_data', under the mother_folder
        # folder_resized is the new folder putting the resized images, os.path.basename() gives the recent folder name
        folder_resized = path.join(folder_final, path.basename(classes_folder) + '_resized')

        # create resized folders
        if not path.exists(folder_resized):
            os.makedirs(folder_resized)

        # resize each of the image in the folder
        for images in os.listdir(path.join(folder, classes_folder)):
            img = cv2.imread(path.join(folder, classes_folder, images), 1)
            img = cv2.resize(img, (width, height))
            cv2.imwrite(path.join(folder_resized, images), img)
        print('Successfully created', folder_resized)

# sample code of using this lib
'''
folder_include_all_classes = 'C:/data/HogwartsHouses/raw_data'
# for folder in os.listdir(mother_folder):
resize_to_somesize(folder_include_all_classes, 32, 32)
'''