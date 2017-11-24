import numpy as np
import tensorflow as tf
import pandas as pd
import os
import time


# MasonicProject
# 2017-7-8-0008
# 使用LSTM模型预测人民的名义收视率

Rootdir = os.path.abspath(os.path.dirname(os.getcwd()))
Modeldir = Rootdir + r"\Models\LSTM\LSTM.model"
Datadir = "E:\PyCharmProjects\MasonicDLv0.1\Database\Renmindemingyi.csv"
Data_Sheet = "Sheet1"

train_step = 3000

learning_rate = tf.train.exponential_decay(
    learning_rate=0.01,
    global_step=train_step,
    decay_steps=100,
    decay_rate=0.9,
    staircase=True)

regularizer_enabled = False
reg_rate = 0.05
hidden_layer_size = 30
seq_size = 10
test_size = 5

X = tf.placeholder(tf.float32, [None, seq_size, 1])
Y = tf.placeholder(tf.float32, [None, 1])

W = {
    'w1': tf.Variable(tf.random_normal([hidden_layer_size, 15])),
    'w2': tf.Variable(tf.random_normal([15, 1])),
    'w3': tf.Variable(tf.random_normal([10, 1])),
    "b1": tf.Variable(tf.random_normal([1])),
    "b2": tf.Variable(tf.random_normal([1])),
    "b3": tf.Variable(tf.random_normal([1]))
}


def normal(data):
    data = (data - data.min()) / (data.max() - data.min())
    return data


def rnn(X, W):
    w1, w2, w3 = W['w1'], W['w2'], W['w3']
    b1, b2, b3 = tf.expand_dims(W['b1'], axis=0), tf.expand_dims(W['b2'], axis=0), W['b3']
    w1 = tf.tile(input=tf.expand_dims(w1, axis=0), multiples=[tf.shape(X)[0], 1, 1])
    w2 = tf.tile(input=tf.expand_dims(w2, axis=0), multiples=[tf.shape(X)[0], 1, 1])
    b1 = tf.tile(input=tf.expand_dims(b1, axis=1), multiples=[tf.shape(X)[0], 1, 1])
    b2 = tf.tile(input=tf.expand_dims(b2, axis=1), multiples=[tf.shape(X)[0], 1, 1])

    cell = tf.nn.rnn_cell.BasicRNNCell(hidden_layer_size)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    fc1 = tf.nn.tanh(tf.matmul(outputs, w1) + b1)
    # y_[batch_size, seq_size, hidden_layer_size]
    fc2 = tf.nn.tanh(tf.matmul(fc1, w2) + b2)
    fc2 = tf.squeeze(fc2)
    y_ = tf.nn.tanh(tf.matmul(fc2, w3) + b3)

    return y_

# def rnn(X, W):
#     cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
#     outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
#     y_ = tf.nn.tanh(tf.matmul(outputs[-1], W['w1']) + W['b1'])
#     y_ = tf.nn.tanh(tf.matmul(y_, W['w2']) + W['b2'])
#     y_ = tf.squeeze(y_)
#
#     return y_


data = pd.read_csv(Datadir, header=None)

data.info()
data = normal(data)
data = np.array(data)
data = data.tolist()
data_size = np.shape(data)[0]
seq, pre = [], []

for i in range(data_size - seq_size - 1):
    seq.append(data[i: i + seq_size])
    pre.append(data[i + seq_size])

data_size = data_size - seq_size - 1
trX = seq[:data_size - 1 - test_size]
trY = pre[:data_size - 1 - test_size]
teX = seq[data_size - 1 - test_size:data_size - 1]
teY = pre[data_size - test_size:]

y_ = rnn(X, W)
loss = tf.reduce_mean(tf.square(Y - y_))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(train_step):
        _, train_loss = sess.run([train_op, loss], feed_dict={X: trX, Y: trY})
        if (step % 100) == 0:
            test_loss = sess.run(loss, feed_dict={X: teX, Y: teY})
            print(step, train_loss, test_loss)
            # if step % 1000 == 0:
                # print(sess.run(y_, feed_dict={X: teX}))
                # print(teY)
