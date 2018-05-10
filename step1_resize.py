import cv2
import os
import os.path as path


# This function resizes images (under the 'folder') to a custom size
def resize_to_somesize(folder, width, height):
    # mother_folder is the upper folder of folder
    mother_folder = path.abspath(folder + '/..')

    # folder_Final is the new folder named 'Final_data', under the mother_folder
    # folder_resized is the new folder putting the resized images, os.path.basename() gives the recent folder name
    folder_Final = path.join(mother_folder, 'Final_data')
    folder_resized = path.join(folder_Final, path.basename(folder) + '_resized')

    # load all the classes folder name
    classes_folder = os.listdir(folder)

    # create 'Final_data' folder
    if not path.exists(folder_Final):
        os.makedirs(folder_Final)
        print('Successfully created', folder_Final)

    # create resized folders
    if not path.exists(folder_resized):
        os.makedirs(folder_resized)

    # resize each of the image in the folder
    for images in classes_folder:
        img = cv2.imread(path.join(folder, images), 1)
        img = cv2.resize(img, (width, height))
        cv2.imwrite(path.join(folder_resized, images), img)
    print('Successfully created', folder_resized)

mother_folder = 'C:/data/HogwartsHouses'
for folder in os.listdir(mother_folder):
    resize_to_somesize(path.join(mother_folder, folder), 50, 50)
