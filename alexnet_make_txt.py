import numpy as np
import os.path as path
import os


def shuffle_and_label(folder, image_width, image_height):
    image_files = os.listdir(folder)
    np.random.shuffle(image_files)
    data_set = np.ndarray(shape=(len(image_files), image_width, image_height, 3), dtype=np.float32)
    label_set = np.ndarray(shape=(len(image_files)), dtype=np.int)
    image_amount = 0
    file = open('test_data.txt', 'w')
    for image in image_files:
        # strange part, when loading into 4d array, it automatically scales up, so need to divide 255
        # label the four houses in Hogwarts (R -> Ravenclaw, G -> Gryffindor, S -> Slytherin, H -> Hufflepuff)
        if image[0] == 'R':
            label_set[image_amount] = 0
        elif image[0] == 'G':
            label_set[image_amount] = 1
        elif image[0] == 'S':
            label_set[image_amount] = 2
        elif image[0] == 'H':
            label_set[image_amount] = 3
        file.write(folder + '/' + image)
        file.write(' ')
        file.write(str(label_set[image_amount]))
        file.write('\n')
        image_amount = image_amount + 1
    file.close()

folder = 'C:/data/HogwartsHouses/dataset_227by227/test_data'
shuffle_and_label(folder, 227, 227)