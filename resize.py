import cv2
import os
import os.path as path

# This function resize images (under the folder argument) to a custom size
def resize_to_size(folder, width, height):
    # mother_folder is the upper folder of folder
    mother_folder = path.abspath(folder + '/..')

    # folder_resized is the new folder putting the resized images, os.path.basename() gives the recent folder name
    folder_resized = path.join(mother_folder, path.basename(folder) + '_resized')

    # create folders
    if not path.exists(folder_resized):
        os.makedirs(folder_resized)

    # resize each of the image in the folder
    for images in os.listdir(folder):
        img = cv2.imread(path.join(folder, images), 1)
        img = cv2.resize(img, (width, height))
        cv2.imwrite(path.join(folder_resized, images), img)

filepath = 'C:/data/HogwartsHouses/gryffindor'
resize_to_size(filepath, 100, 100)
