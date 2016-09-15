# IPython log file

get_ipython().magic(u'logstart bw_rgb.py')
import tensorflow as tf
x_raw = tf.placeholder(tf.float32, shape=(None, 1024*3, 1024*2, 3))
x_bw = tf.image.rgb_to_grayscale(x_raw)
def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

h_conv1 = tf.nn.relu(conv2d(x_bw, W_conv1) + b_conv1)
W_conv1 = tf.Variable(tf.zeros([5, 5, 1, 32]))
b_conv1 = tf.Variable(tf.zeros([32]))
h_conv1 = tf.nn.relu(conv2d(x_bw, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = tf.Variable(tf.zeros([5, 5, 32, 64]))
b_conv2 = tf.Variable(tf.zeros([64]))
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
w_d = tf.Variable(tf.zeros([768*512*64,1024*3*1024*2*3]))
b_d = tf.Variable(tf.zeros([1024*3*1024*2*3]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 768*512*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_d) + b_d)
y_h = tf.reshape(h_fc1,[-1,1024*3,1024*2,3])
