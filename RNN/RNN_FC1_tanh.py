import tensorflow as tf
import pandas as pd
import numpy as np

step = 150

seq_size = 10
vector_size = 1
batch_size = 10
test_size = 5

train_percent = 0.7

X = tf.placeholder("float", [None, seq_size, vector_size])
Y = tf.placeholder("float", [None, 1])

W = {
    "w1": tf.Variable(tf.random_normal([10, 1])),
    "b1": tf.Variable(tf.random_normal([1]))
}

data = pd.read_csv("E:\PyCharmProjects\MasonicDLv0.1\Database\无证之罪.csv", header=None)
data = data[1]

# Normalization
data = (data - data.min()) / (data.max() - data.min())

x, y = [], []
for i in range(len(data) - seq_size):
    x.append(np.expand_dims(data[i:i + seq_size], axis=1))
    y.append(np.expand_dims(data[i + seq_size], axis=0))

tr_num = int(train_percent * len(data))

trX, trY = x[:tr_num], y[:tr_num]
teX, teY = x[tr_num:], y[tr_num:]


def nn(X, W, seq_size, vector_size):
    X = tf.transpose(X, [1, 0, 2])
    X = tf.reshape(X, [-1, vector_size])
    X = tf.split(X, seq_size, 0)
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=10)
    outputs, _ = tf.nn.static_rnn(cell, X, dtype=tf.float32)
    y_ = tf.nn.tanh(tf.matmul(outputs[-1], W["w1"]) + W["b1"])

    return y_


y_ = nn(X, W, seq_size, vector_size)
loss = tf.reduce_mean(tf.square(Y - y_))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(step):
        train_loss, test_loss = 0, 0
        for end in range(batch_size, len(trY), batch_size):
            begin = end - batch_size
            x = trX[begin:end]
            y = trY[begin:end]
            train_loss, _ = sess.run([loss, train_op], feed_dict={X: x, Y: y})
            test_indices = np.arange(len(teX))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
            trX_sliced, trY_sliced = [], []
            for t in test_indices:
                trX_sliced.append(trX[t])
                trY_sliced.append(trY[t])
            test_loss = sess.run(loss, feed_dict={X: trX_sliced, Y: trY_sliced})
        print("Train Step: ", i)
        print("Accuracy on train/test: %f/%f" % (np.mean(train_loss), np.mean(test_loss)))
