import tensorflow as tf
from six.moves import cPickle as pickle
import numpy as np
from scipy import ndimage


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    # truncated normal distribution
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [Batch, Height, Width, Channel], in the computer's point if view, it sees just four dimensions
    # so we shouldn't pass through any of samples(batch) or channels
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [Batch, Height, Width, Channel]
    # ksize [Batch, Height, Width, Channel]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 100, 100], name='x_input')   # 100*100
ys = tf.placeholder(tf.float32, [None, 4], name='y_input')
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 100, 100, 1])
# print(x_image.shape)  # [n_samples, 100,100]

# define the depth of each layer
d_cnn1 = 4
d_cnn2 = 4

## conv1 layer ##
W_conv1 = weight_variable([5, 5, 1, d_cnn1])  # patch 5x5, in size 1, out size 8
b_conv1 = bias_variable([d_cnn1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 100x100x8
h_pool1 = max_pool_2x2(h_conv1)                           # output size 50x50x8

## conv2 layer ##
W_conv2 = weight_variable([5, 5, d_cnn1, d_cnn2])  # patch 5x5, in size 8, out size 16
b_conv2 = bias_variable([d_cnn2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 50x50x16
h_pool2 = max_pool_2x2(h_conv2)                           # output size 25x25x16

## fc1 layer ##
W_fc1 = weight_variable([25*25*d_cnn2, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 25, 25, 16] ->> [n_samples, 25*25*16]
h_pool2_flat = tf.reshape(h_pool2, [-1, 25*25*d_cnn2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 4])
b_fc2 = bias_variable([4])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data, two kinds of cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
mse = tf.reduce_mean(0.5 * tf.square(ys - prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)
# saver = tf.train.Saver()
# saver.restore(sess, "my_net/save_net.ckpt")

with open('./train_data/00_data.pickle', 'rb') as f:
    tr_dat = pickle.load(f)
with open('./train_data/00_label.pickle', 'rb') as f:
    tr_lab = pickle.load(f)
with open('./test_data/00_data.pickle', 'rb') as f:
    te_dat = pickle.load(f)
with open('./test_data/00_label.pickle', 'rb') as f:
    te_lab = pickle.load(f)

# # building ndarrays for storing results after filters
# tr_dat_after_sobel = np.ndarray(shape=(np.shape(tr_dat)), dtype=np.float32)
# tr_dat_after_prewitt = np.ndarray(shape=(np.shape(tr_dat)), dtype=np.float32)
# tr_dat_after_laplacian = np.ndarray(shape=(np.shape(tr_dat)), dtype=np.float32)
# tr_dat_after_gaussian_laplace = np.ndarray(shape=(np.shape(tr_dat)), dtype=np.float32)
#
# te_dat_after_sobel = np.ndarray(shape=(np.shape(te_dat)), dtype=np.float32)
# te_dat_after_prewitt = np.ndarray(shape=(np.shape(te_dat)), dtype=np.float32)
# te_dat_after_laplacian = np.ndarray(shape=(np.shape(te_dat)), dtype=np.float32)
# te_dat_after_gaussian_laplace = np.ndarray(shape=(np.shape(te_dat)), dtype=np.float32)
#
# # filter operations on training data
# for _ in range(320):
#     tr_dat_after_sobel[_, :, :] = ndimage.sobel(tr_dat[_, :, :], 0)
#     tr_dat_after_prewitt[_, :, :] = ndimage.prewitt(tr_dat[_, :, :], 0)
#     tr_dat_after_laplacian[_, :, :] = ndimage.laplace(tr_dat[_, :, :])
#     tr_dat_after_gaussian_laplace[_, :, :] = ndimage.gaussian_laplace(tr_dat[_, :, :], sigma=1)
#
# # filter operations on test data
# for _ in range(160):
#     te_dat_after_sobel[_, :, :] = ndimage.sobel(te_dat[_, :, :], 0)
#     te_dat_after_prewitt[_, :, :] = ndimage.prewitt(te_dat[_, :, :], 0)
#     te_dat_after_laplacian[_, :, :] = ndimage.laplace(te_dat[_, :, :])
#     te_dat_after_gaussian_laplace[_, :, :] = ndimage.gaussian_laplace(te_dat[_, :, :], sigma=1)

# training process starts
batch_size = 200
for epoch in range(300):       # epoch amount
    for batch in range(len(tr_dat) // batch_size):
        sess.run(train_step, feed_dict={xs: tr_dat[batch * batch_size: (batch + 1) * batch_size],
                                        ys: tr_lab[batch * batch_size: (batch + 1) * batch_size], keep_prob: 0.5})
    if epoch % 10 == 0:
        print(epoch, 'th', compute_accuracy(te_dat, te_lab))

sess.close()

