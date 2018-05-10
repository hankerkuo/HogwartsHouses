import matplotlib.pyplot as plt
import cv2
from six.moves import cPickle as pickle
import numpy as np

# extract data
with open('./train_data/00_data.pickle', 'rb') as f:
    tr_dat = pickle.load(f)
with open('./train_data/00_label.pickle', 'rb') as f:
    tr_lab = pickle.load(f)
with open('./test_data/00_data.pickle', 'rb') as f:
    te_dat = pickle.load(f)
with open('./test_data/00_label.pickle', 'rb') as f:
    te_lab = pickle.load(f)

# image = np.ndarray(shape=(600, 100, 100))
# image_small = np.ndarray(shape=(100, 100, 3))
# image[0] = (cv2.imread('./train_data/Gryffindor_004.jpg'))/255
print(tr_dat[40])
# print(tr_lab[40])
cv2.imshow('Houses', tr_dat[90])
cv2.waitKey(0)
