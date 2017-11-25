import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

from Utility.Normalize import mnormalize
from Utility.XlsReader import readxlsbycol

# 以白鹿原播放量和微博热度为数据库构建的双变量单层LSTM模型

Datadir = "E:\PyCharmProjects\MasonicDLv0.1\Database\白鹿原_MIX.xlsx"
Data_Sheet = "白鹿原_MIX"


train_step = 10000
global_step = tf.Variable(0, name="global_step")
learning_rate = tf.train.exponential_decay(learning_rate=0.01,
                                           global_step=global_step,
                                           decay_steps=100,
                                           decay_rate=0.9,
                                           staircase=True)
regularizer_enabled = False
reg_rate = 0.5
hidden_layer_size = 30
seq_size = 10
keep_prob = 0.9
W = {
    'q_w1': tf.Variable(tf.random_normal([hidden_layer_size, 1]))
}
X = tf.placeholder(tf.float32, [None, seq_size, 2])
Y = tf.placeholder(tf.float32, [None, seq_size])


def rnn(X, W):
    q_w1 = W['q_w1']
    # q_w1(30, 1)
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_layer_size)

    outputs, states = tf.nn.dynamic_rnn(cell, inputs=X, dtype=tf.float32, time_major=False)
    # outputs(?, 10, 30)
    # X1(?, 10, 1)
    multi = [tf.shape(X)[0], 1, 1]
    # multi(?, 1, 1)
    q_w1_i = tf.expand_dims(q_w1, 0)
    # (1, 30 ,1)
    q_w2 = tf.tile(input=q_w1_i, multiples=multi)
    # (?, 30, 1)
    # 通过tile方法共享参数

    # 此处添加偏置项不当可能导致图像整体平移
    # b = tf.Variable(tf.random_normal([1]), name='b')
    y_ = tf.nn.tanh(tf.matmul(outputs, q_w2))
    # (?, 10, 30)*(?, 30, 1) = (?, 10, 1)
    # <==> (10, 30)*(30, 1) = (10, 1)

    y_ = tf.squeeze(y_)
    return y_, q_w1, q_w2


qiyi_f = readxlsbycol(Datadir, Data_Sheet, 1)
sina_f = readxlsbycol(Datadir, Data_Sheet, 2)

q_X = mnormalize(qiyi_f)
s_X = mnormalize(sina_f)

q_seq, s_seq, y_seq = [], [], []
for i in range(len(q_X) - seq_size - 1):
    q_seq.append(np.expand_dims(q_X[i:i + seq_size], axis=1).tolist())
    s_seq.append(np.expand_dims(s_X[i:i + seq_size], axis=1).tolist())
    y_seq.append(q_X[i + 1:i + seq_size + 1].tolist())

q_trX = q_seq[:151]
q_teX = q_seq[151:]
s_trX = s_seq[:151]
s_teX = s_seq[151:]
trY = y_seq[:151]

trX = np.concatenate((q_trX, s_trX), axis=2)
teX = np.concatenate((q_teX, s_teX), axis=2)

y_, w1, w2 = rnn(X, W)

if regularizer_enabled:
    loss = tf.reduce_mean(tf.square(Y - y_)) + \
           tf.contrib.layers.l2_regularizer(reg_rate)(w1) + \
           tf.contrib.layers.l2_regularizer(reg_rate)(w2)
else:
    loss = tf.reduce_mean(tf.square(y_ - Y))

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(train_step):
        _, test_loss = sess.run([train_op, loss], feed_dict={X: trX, Y: trY})
        if step % 100 == 0:
            print(step, test_loss)
