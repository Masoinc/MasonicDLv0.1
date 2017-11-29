import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt_title = "DNN"
step = 5000

seq_size = 10
vector_size = 2
batch_size = 30
test_size = 30

global_step = tf.Variable(0, name="global_step")
learning_rate = tf.train.exponential_decay(
    learning_rate=0.01,
    global_step=global_step,
    decay_steps=100,
    decay_rate=0.9,
    staircase=True)

train_percent = 0.9

reg = True
reg_rate = 0.0005

X = tf.placeholder("float", [None, seq_size, vector_size])
Y = tf.placeholder("float", [None, 1])

W = {
    "w1": tf.Variable(tf.random_normal([10, 30])),
    "w2": tf.Variable(tf.random_normal([30, 15])),
    "w3": tf.Variable(tf.random_normal([15, 10])),
    "w4": tf.Variable(tf.random_normal([10, 1])),
    "b1": tf.Variable(tf.random_normal([1])),
    "b2": tf.Variable(tf.random_normal([1])),
    "b3": tf.Variable(tf.random_normal([1])),
    "b4": tf.Variable(tf.random_normal([1]))
}


def normal(data):
    data = (data - data.min()) / (data.max() - data.min())
    return data


def get_seq():
    data = pd.read_csv("G:\Idea\MasonicDLv0.1\Database\MultiDBpartI_noheader.csv", header=None,
                       skip_blank_lines=True)

    x, y = [], []
    for i in range(0, 10, 2):
        qi, si = normal(data[i]), normal(data[i + 1])
        qi, si = qi.dropna(axis=0, how='all'), si.dropna(axis=0, how='all')
        for k in range(len(qi) - seq_size):
            # [10, 10]
            x.append(np.transpose([qi[k:k + seq_size], si[k:k + seq_size]]))
            y.append(np.expand_dims(qi[k + seq_size], axis=0))
    return x, y


x, y = get_seq()
tr_num = int(train_percent * len(y))

trX, trY = x[:tr_num], y[:tr_num]
teX, teY = x[tr_num:], y[tr_num:]


def nn(X, W, seq_size, vector_size):
    # X = tf.transpose(X, [1, 0, 2])
    # X = tf.reshape(X, [-1, vector_size])
    # X = tf.split(X, seq_size, 0)
    # cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10)
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=10)
    # outputs, _ = tf.nn.static_rnn(cell, X, dtype=tf.float32)
    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    # y_ = tf.nn.tanh(tf.matmul(outputs[-1], W["w1"]) + W["b1"])
    #
    # y_ = tf.nn.tanh(tf.matmul(y_, W["w2"]) + W["b2"])
    #
    # y_ = tf.nn.tanh(tf.matmul(y_, W["w3"]) + W["b3"])
    #
    # y_ = tf.nn.tanh(tf.matmul(y_, W["w4"]) + W["b4"])
    b1, b2, b3, b4 = tf.expand_dims(W['b1'], axis=0), \
                     tf.expand_dims(W['b2'], axis=0), \
                     tf.expand_dims(W['b3'], axis=0), \
                     tf.expand_dims(W['b4'], axis=0)
    w1, w2, w3, w4 = tf.expand_dims(W['w1'], axis=0), \
                     tf.expand_dims(W['w2'], axis=0), \
                     tf.expand_dims(W['w3'], axis=0), tf.expand_dims(W['w4'], axis=0)
    W_ = [W['w1'], W['w2'], W['w3'], W['w4']]
    # y_ = tf.nn.tanh(tf.matmul(outputs, W["w1"]) + W["b1"])
    w1 = tf.tile(input=w1, multiples=[tf.shape(outputs)[0], 1, 1])
    w2 = tf.tile(input=w2, multiples=[tf.shape(outputs)[0], 1, 1])
    w3 = tf.tile(input=w3, multiples=[tf.shape(outputs)[0], 1, 1])
    w4 = tf.tile(input=w4, multiples=[tf.shape(outputs)[0], 1, 1])
    fc1 = tf.nn.tanh(tf.matmul(outputs, w1) + b1)

    fc2 = tf.nn.tanh(tf.matmul(fc1, w2) + b2)

    fc3 = tf.nn.tanh(tf.matmul(fc2, w3) + b3)

    fc4 = tf.nn.tanh(tf.matmul(fc3, w4) + b4)
    y_ = tf.unstack(fc4, axis=1)
    y_ = tf.squeeze(y_)[-1]
    return y_, W_


y_, W_ = nn(X, W, seq_size, vector_size)
if reg:
    loss = tf.reduce_mean(tf.square(Y - y_)) + \
           tf.contrib.layers.l2_regularizer(reg_rate)(W_[0]) + \
           tf.contrib.layers.l2_regularizer(reg_rate)(W_[1]) + \
           tf.contrib.layers.l2_regularizer(reg_rate)(W_[2]) + \
           tf.contrib.layers.l2_regularizer(reg_rate)(W_[3])
else:
    loss = tf.reduce_mean(tf.square(Y - y_))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(step):
        train_loss, test_loss = 0, 0
        loss_sum = 0
        for end in range(batch_size, len(trY), batch_size):
            begin = end - batch_size
            x = trX[begin:end]
            y = trY[begin:end]
            train_loss, _ = sess.run([loss, train_op], feed_dict={X: x, Y: y})
            loss_sum += train_loss
        if i % 100 == 0 and i > 1:
            test_indices = np.arange(len(teX))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
            trX_sliced, trY_sliced = [], []
            for t in test_indices:
                trX_sliced.append(teX[t])
                trY_sliced.append(teY[t])
            test_loss = sess.run(loss, feed_dict={X: trX_sliced, Y: trY_sliced})
            print("Train Step: ", i)
            print("Accuracy on train/test: %f/%f" % (
                pow(loss_sum / test_size, 0.5), pow(test_loss, 0.5)))
            if i % 1000 == 0:
                preY = sess.run(y_, feed_dict={X: trX_sliced})
                realY = np.array(trY_sliced)
                realY = np.squeeze(realY)
                x_axis = np.arange(0, 22)

                plt.figure(figsize=(16, 10))
                ax = plt.gca()
                ax.xaxis.grid(True)
                ax.set_xticks(x_axis)
                # plt.vlines(data_size - test_size, 0, 1, colors="c", linestyles="dashed", label='train/test split')
                plt.scatter(x_axis, preY, label="Prediction")
                plt.scatter(x_axis, realY, label="Observation")
                plt.legend()
                plt.title(plt_title)
                plt.show()
