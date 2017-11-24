import tensorflow as tf
import pandas as pd
import numpy as np

Datadir = "E:\PyCharmProjects\MasonicDLv0.1\Database\\"
step = 300

vector_size = 2
batch_size = 50
test_size = 50

rnn_size = 15
conv1_size = 5
conv2_size = 5

seq_size = 15

train_percent = 0.7

learning_rate = tf.train.exponential_decay(
    learning_rate=0.01,
    global_step=step,
    decay_steps=30,
    decay_rate=0.9,
    staircase=True)

X = tf.placeholder("float", [None, seq_size, vector_size])
Y = tf.placeholder("float", [None, 1])

W = {
    "w1": tf.Variable(tf.random_normal([rnn_size, 30])),
    "w2": tf.Variable(tf.random_normal([30, 15])),
    "w3": tf.Variable(tf.random_normal([15, 5])),
    "w4": tf.Variable(tf.random_normal([5, 1])),
    "w5": tf.Variable(tf.random_normal([7, 1])),
    "b1": tf.Variable(tf.random_normal([1])),
    "b2": tf.Variable(tf.random_normal([1])),
    "b3": tf.Variable(tf.random_normal([1])),
    "b4": tf.Variable(tf.random_normal([1])),
    "b5": tf.Variable(tf.random_normal([1])),
    "conv1": tf.Variable(tf.random_normal([conv1_size, 1, 1, 2])),
    "conv1b": tf.Variable(tf.random_normal([1, 2])),
    "conv2": tf.Variable(tf.random_normal([conv2_size, 1, 1, 2])),
    "conv2b": tf.Variable(tf.random_normal([1, 2]))
}


def normal(data):
    data = (data - data.min()) / (data.max() - data.min())
    return data


def get_seq():
    data = pd.read_csv("E:\PyCharmProjects\MasonicDLv0.1\Database\MultiDB_noheader.csv", header=None,
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


# TODO: Tensorflow does not support unstack a tensor with a dimension is ?(None)
# TODO: Function tf.unstack(X, axis=Y), the Y must be specified
# TODO: Trying to rewrite the unstack procedure

def nn(X, W, seq_size, vector_size):
    # X[batch_size=50, seq_size=15, vector_size=2]
    # conv
    seq_size_afterconv = seq_size - conv1_size + 1 - conv2_size + 1
    conv = []
    for i in range(vector_size):
        Vec_Slice = tf.unstack(X, axis=2)[i]  # [None, 15]
        conv_per_batch = []
        for k in range(batch_size):
            Batch_Slice = Vec_Slice[k]  # [15]
            Batch_Slice = tf.expand_dims(Batch_Slice, 0)  # [1, 15]
            Batch_Slice = tf.expand_dims(Batch_Slice, 2)  # [1, 15, 1]

            # Conv Layer1
            conv1 = tf.unstack(W['conv1'], axis=3)
            b = tf.unstack(W['conv1b'], axis=1)
            conv1 = tf.nn.tanh(tf.nn.conv1d(Batch_Slice, conv1[i], stride=1, padding="VALID") + b[i])
            # conv1 = tf.nn.dropout(conv1, keep_prob=0.9)
            conv1 = tf.expand_dims(conv1, axis=2)
            # conv1[1, 11, 1]
            conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 1, 1], strides=[1, 1, 1, 1], padding="SAME")
            conv1 = tf.squeeze(conv1, axis=3)
            # Conv Layer2
            conv2 = tf.unstack(W['conv2'], axis=3)
            b = tf.unstack(W['conv2b'], axis=1)
            conv2 = tf.nn.tanh(tf.nn.conv1d(conv1, conv2[i], stride=1, padding="VALID") + b[i])
            conv2 = tf.expand_dims(conv2, axis=2)
            conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 1, 1], strides=[1, 1, 1, 1], padding="SAME")
            conv2 = tf.squeeze(conv2, axis=3)
            # conv2[1, 7, 1]

            conv_per_batch.append(conv2)  # 50x[1, 7, 1]
        conv_per_batch = tf.concat(conv_per_batch, 0)  # [50, 7, 1]
        conv.append(conv_per_batch)  # 2x[50, 7, 1]
    X = tf.concat(conv, 2)

    # X = tf.transpose(X, [1, 0, 2])
    # # X[seq_size=7, batch_size=50,  vector_size=2]
    # X = tf.reshape(X, [-1, vector_size])
    # # X[seq_size=7*batch_size=50,  vector_size=2]
    # X = tf.split(X, seq_size_afterconv, axis=0)
    # # X[seq_size=7*batch_size=50,  vector_size=2]
    # X[array[batch_size,2]...x7]
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size)

    outputs, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    # outputs[batch_size, 7, 15]
    b1, b2, b3, b4 = tf.expand_dims(W['b1'], axis=0), \
                     tf.expand_dims(W['b2'], axis=0), \
                     tf.expand_dims(W['b3'], axis=0), \
                     tf.expand_dims(W['b4'], axis=0)
    w1, w2, w3, w4 = tf.expand_dims(W['w1'], axis=0), \
                     tf.expand_dims(W['w2'], axis=0), \
                     tf.expand_dims(W['w3'], axis=0), tf.expand_dims(W['w4'], axis=0)

    w1 = tf.tile(input=w1, multiples=[tf.shape(outputs)[0], 1, 1])
    w2 = tf.tile(input=w2, multiples=[tf.shape(outputs)[0], 1, 1])
    w3 = tf.tile(input=w3, multiples=[tf.shape(outputs)[0], 1, 1])
    w4 = tf.tile(input=w4, multiples=[tf.shape(outputs)[0], 1, 1])
    # outputs[50, 7, 15]
    fc1 = tf.nn.relu6(tf.matmul(outputs, w1) + b1)
    # fc1[50, 7, 30]
    fc2 = tf.nn.relu6(tf.matmul(fc1, w2) + b2)
    # fc1[50, 30, 20]
    fc3 = tf.nn.relu6(tf.matmul(fc2, w3) + b3)
    # fc1[50, 20, 10]
    fc4 = tf.nn.relu6(tf.matmul(fc3, w4) + b4)
    # fc1[50, 10, 1]
    fc5 = tf.squeeze(fc4)
    # y_[50, 10]
    y_ = tf.nn.relu6(tf.matmul(fc5, W['w5']) + W['b5'])

    return y_  # [50,1]


y_ = nn(X, W, seq_size, vector_size)
loss = tf.reduce_mean(tf.square(Y - y_))
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(step):
        train_loss, test_loss = 0, 0
        for end in range(batch_size, len(trY), batch_size):
            # Train
            begin = end - batch_size
            x = trX[begin:end]
            y = trY[begin:end]
            train_loss, _ = sess.run([loss, train_op], feed_dict={X: x, Y: y})

        if i % 5 == 0:
            # Test loss
            test_indices = np.arange(len(teX))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
            trX_sliced, trY_sliced = [], []
            for t in test_indices:
                trX_sliced.append(teX[t])
                trY_sliced.append(teY[t])
            test_loss = sess.run(loss, feed_dict={X: trX_sliced, Y: trY_sliced})
            print("Train Step: ", i)
            print("Accuracy on train/test: %f/%f" % (np.mean(train_loss), np.mean(test_loss)))
            # print("Accuracy on train: %f" % (np.mean(train_loss)))
