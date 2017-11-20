import pandas as pd
import Utility.Sequencer
import tensorflow as tf
import numpy as np

datas = Utility.Sequencer.get_raw()
# print(np.shape(datas))
# print(datas[0][0])
# [143, 2]

step = 1000

drama_num = 12
Conv1_Core = 5
Conv2_Core = 5
k_max1 = 60
k_max2 = 45

X = tf.placeholder("float", [None, 2])
W = {
    "conv1_core": tf.Variable(tf.random_normal([Conv1_Core, 1, 1, 2])),
    "conv1_b": tf.Variable(tf.random_normal([1, 2])),
    "conv2_core": tf.Variable(tf.random_normal([Conv2_Core, 1, 1, 2])),
    "conv2_b": tf.Variable(tf.random_normal([1, 2]))
}


# 输入为不定长batch, 每个batch为一部电视剧

def k_max(X, W):
    # turn to slice
    conv = []
    for i in range(X.shape[1]):
        X_Slice = tf.unstack(X, axis=1)[i]  # [?]
        X_Slice = tf.expand_dims(X_Slice, 0)  # [1, ?]
        X_Slice = tf.expand_dims(X_Slice, 2)  # [1, ?, 1]

        # Conv.1
        conv1_core = tf.unstack(W['conv1_core'], axis=3)
        conv1_b = tf.unstack(W['conv1_b'], axis=1)
        conv1 = tf.nn.tanh(
            tf.nn.conv1d(X_Slice, conv1_core[i], stride=1, padding="SAME") + conv1_b[i])  # [1, ?, 1]

        # Max-k Pool.1
        X_Slice = tf.reshape(conv1, [1, 1, -1])
        v = tf.nn.top_k(X_Slice, k=k_max1, sorted=False).values

        conv2_core = tf.unstack(W['conv2_core'], axis=3)
        conv2_b = tf.unstack(W['conv2_b'], axis=1)
        conv2 = tf.nn.tanh(tf.nn.conv1d(v, conv2_core[i], stride=1, padding="SAME") + conv2_b[i])

        X_Slice = tf.reshape(conv2, [0, 2, 1])
        v = tf.nn.top_k(X_Slice, k=k_max2, sorted=False).values

        conv.append(v)
    convs = tf.stack(conv, axis=2)
    return convs


def lstm(X, W):
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10)


# pool1 = tf.nn.max_pool(X_Slice, ksize=[1, k, 1, 1], strides=[1, 1, 1, 1], padding="SAME")

# TODO: 写出按照大小排序的真 K-Max 池化操作
# TODO: 卷积时升采样，随后进行合并

conv = k_max(X, W)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(datas[0].shape[0])
    v = sess.run(conv, feed_dict={X: datas[0]})

    # print(conv)
    # for i in range(step):
    #     train_loss, test_loss = 0, 0
    #     # A drama is a batch
    #     for k in range(drama_num):
    #         data = datas[k]
    #         conv = k_max(data, W)
