import matplotlib.pyplot as plt
import cv2
from six.moves import cPickle as pickle
import numpy as np

# extract data
image_width = 32
image_height = 32
dataset_path = 'C:/data/HogwartsHouses/dataset_%dby%d' % (image_width, image_height)
with open(dataset_path + '/train_data/00_data.pickle', 'rb') as f:
    tr_dat = pickle.load(f)
with open(dataset_path + '/train_data/00_label.pickle', 'rb') as f:
    tr_lab = pickle.load(f)
with open(dataset_path + '/test_data/00_data.pickle', 'rb') as f:
    te_dat = pickle.load(f)
with open(dataset_path + '/test_data/00_label.pickle', 'rb') as f:
    te_lab = pickle.load(f)

train_x, train_y, test_x, test_y = tr_dat, tr_lab, te_dat, te_lab
# image = np.ndarray(shape=(600, 100, 100))
# image_small = np.ndarray(shape=(100, 100, 3))
# image[0] = (cv2.imread('./train_data/Gryffindor_004.jpg'))/255
print(tr_dat[0])
# print(tr_lab[40])
cv2.imshow('Houses', tr_dat[0])
cv2.waitKey(0)
