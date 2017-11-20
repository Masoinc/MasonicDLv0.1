import numpy as np
import tensorflow as tf
import pandas as pd
import os

# MasonicProject
# 2017-7-8-0008
# 使用LSTM模型预测人民的名义收视率
# 及获取RMSE等拟合优度参数


# 加载数据

Rootdir = os.path.abspath(os.path.dirname(os.getcwd()))
Modeldir = Rootdir + r"\Models\LSTM\LSTM.model"
Datadir = "E:\PyCharmProjects\MasonicDLv0.1\Database\Renmindemingyi.csv"
Data_Sheet = "Sheet1"

train_step = 5000

learning_rate = tf.train.exponential_decay(
    learning_rate=0.001,
    global_step=train_step,
    decay_steps=100,
    decay_rate=0.9,
    staircase=True)

regularizer_enabled = False
reg_rate = 0.05
hidden_layer_size = 15
seq_size = 10

X = tf.placeholder(tf.float32, [None, seq_size, 1])
Y = tf.placeholder(tf.float32, [None, seq_size])

W = {
    'w1': tf.Variable(tf.random_normal([hidden_layer_size, 1])),
    "b1": tf.Variable(tf.random_normal([1]))
}


def normal(data):
    data = (data - data.min()) / (data.max() - data.min())
    return data


def rnn(X, W):
    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    w2 = tf.tile(tf.expand_dims(W['w1'], 0), [tf.shape(X)[0], 1, 1])
    y_ = tf.nn.tanh(tf.matmul(outputs, w2) + W['b1'])
    y_ = tf.squeeze(y_)

    return y_


data = pd.read_csv(Datadir, header=None)
data.info()
data = normal(data)
tr = data[:-5]

trX, trY, = [], []
for i in range(len(tr) - seq_size - 1):
    trX.append(np.expand_dims(tr[i: i + seq_size], axis=1))
    trY.append(tr[i + 1: i + seq_size + 1])

y_ = rnn(X, W)
loss = tf.reduce_mean(tf.square(Y - y_))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(train_step):
        _, loss_ = sess.run([train_op, loss], feed_dict={X: trX, Y: trY})
        if step % 100 == 0:
            print(step, loss_)

            # if regularizer_enabled:
            #     loss = tf.reduce_mean(tf.square(Y - y_)) + \
            #            tf.contrib.layers.l2_regularizer(reg_rate)(w1) + \
            #            tf.contrib.layers.l2_regularizer(reg_rate)(w2)
            # else:
