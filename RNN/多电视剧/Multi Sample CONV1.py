import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Datadir = "E:\PyCharmProjects\MasonicDLv0.1\Database\MultiDBpartI_noheader.csv"
Modeldir = "E:\PyCharmProjects\MasonicDLv0.1\Models\RNN\Conv\\CLDNN.model"

plt_title = "CONV2_POOL"

with tf.name_scope(name='Hyperparameter'):
    step = 5001
    seq_size = 15
    vector_size = 2
    batch_size = 50
    test_size = 50

    rnn_size = 10
    conv1_size = 3

    reg = False
    reg_rate = 0.002

    train_percent = 0.7
    global_step = tf.Variable(0, name="global_step")
    learning_rate = tf.train.exponential_decay(
        learning_rate=0.1,
        global_step=global_step,
        decay_steps=80,
        decay_rate=0.9,
        staircase=True)

with tf.name_scope(name='Placeholder'):
    X = tf.placeholder("float", [None, seq_size, vector_size])
    Y = tf.placeholder("float", [None, ])

    W = {
        "w1": tf.Variable(tf.random_normal([rnn_size, 20])),
        "w2": tf.Variable(tf.random_normal([20, 15])),
        "w3": tf.Variable(tf.random_normal([15, 5])),
        "w4": tf.Variable(tf.random_normal([5, 1])),
        "b1": tf.Variable(tf.random_normal([1])),
        "b2": tf.Variable(tf.random_normal([1])),
        "b3": tf.Variable(tf.random_normal([1])),
        "b4": tf.Variable(tf.random_normal([1])),
        "conv1": tf.Variable(tf.random_normal([conv1_size, 1, 1, 2])),
        "conv1b": tf.Variable(tf.random_normal([1, 2]))
    }

with tf.name_scope(name='DataProcessing'):
    def normal(data):
        data = (data - data.min()) / (data.max() - data.min())
        return data


    def get_seq():
        data = pd.read_csv(Datadir, header=None,
                           skip_blank_lines=True)
        x, y = [], []
        for i in range(0, 10, 2):
            qi, si = normal(data[i]), normal(data[i + 1])
            qi, si = qi.dropna(axis=0, how='all'), si.dropna(axis=0, how='all')
            for k in range(len(qi) - seq_size + 1 - 1):
                # [10, 10]
                x.append(np.transpose([qi[k:k + seq_size], si[k:k + seq_size]]))
                y.append(qi[k + seq_size])
        return x, y


    x, y = get_seq()

    tr_num = int(train_percent * len(y))
    trX, trY = x[:tr_num], y[:tr_num]
    teX, teY = x[tr_num:], y[tr_num:]

# TODO: Tensorflow does not support unstack a tensor with a dimension is ?(None)
# TODO: Function tf.unstack(X, axis=Y), the Y must be specified
# TODO: Trying to rewrite the unstack procedure
with tf.name_scope(name='NeuralNetwork'):
    def nn(X, W, seq_size, vector_size):
        conv = []
        seq_size_afterconv = seq_size - conv1_size + 1
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
                # conv1 = tf.expand_dims(conv1, axis=2)
                # conv1 = tf.nn.dropout(conv1, keep_prob=0.9)
                # conv1[1, 11, 1]
                # conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 1, 1], strides=[1, 1, 1, 1], padding="SAME")
                # conv1 = tf.squeeze(conv1, axis=3)

                conv_per_batch.append(conv1)  # 50x[1, 7, 1]
            conv_per_batch = tf.concat(conv_per_batch, 0)  # [50, 7, 1]
            conv.append(conv_per_batch)  # 2x[50, 7, 1]
        X = tf.concat(conv, 2)
        # X[50, 7, 2]
        X1 = tf.transpose(X, [1, 0, 2])
        # X1[7, 50, 2]
        X2 = tf.reshape(X1, [-1, vector_size])
        # X2[7*50, 2]
        X3 = tf.split(X2, seq_size_afterconv, axis=0)
        # X3[(batch_size, 2), ..., seq_size_afterconvx]
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size)

        outputs, _ = tf.nn.static_rnn(cell, X3, dtype=tf.float32)

        fc1 = tf.nn.tanh(tf.matmul(outputs[-1], W['w1']) + W['b1'])
        # fc1[50, 7, 30]
        fc2 = tf.nn.tanh(tf.matmul(fc1, W['w2']) + W['b2'])
        # fc1[50, 30, 20]
        fc3 = tf.nn.tanh(tf.matmul(fc2, W['w3']) + W['b3'])
        # fc1[50, 20, 10]
        fc4 = tf.nn.tanh(tf.matmul(fc3, W['w4']) + W['b4'])
        # fc1[50, 10, 1]
        y_ = tf.squeeze(fc4)
        W_ = [W['w1'], W['w2'], W['w3'], W['w4'], W['conv1']]
        # y_[50, 10]
        # y_ = tf.nn.tanh(tf.matmul(fc5, W['w5']) + W['b5'])

        return y_, W_  # [50,1]

with tf.name_scope(name='TrainSettings'):
    y_, W_ = nn(X, W, seq_size, vector_size)
    if reg:
        loss = tf.reduce_mean(tf.square(Y - y_)) + \
               tf.contrib.layers.l2_regularizer(reg_rate)(W_[0]) + \
               tf.contrib.layers.l2_regularizer(reg_rate)(W_[1]) + \
               tf.contrib.layers.l2_regularizer(reg_rate)(W_[2]) + \
               tf.contrib.layers.l2_regularizer(reg_rate)(W_[3]) + \
               tf.contrib.layers.l2_regularizer(reg_rate)(W_[4])
    else:
        loss = tf.reduce_mean(tf.square(Y - y_))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    saver = tf.train.Saver(tf.global_variables())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(step):
        train_loss, test_loss = 0, 0
        loss_sum = 0
        for end in range(batch_size, len(trY), batch_size):
            # Train
            begin = end - batch_size
            x = trX[begin:end]
            y = trY[begin:end]
            train_loss, _ = sess.run([loss, train_op], feed_dict={X: x, Y: y})
            loss_sum += train_loss
        if i % 100 == 0 and i > 1:
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
            print("Accuracy on train/test: %f/%f" % (
                pow(loss_sum / test_size, 0.5), pow(test_loss, 0.5)))

            if i == step - 1:
                preY = sess.run(y_, feed_dict={X: trX_sliced})
                realY = np.array(trY_sliced)

                x_axis = np.arange(0, test_size)

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

    print("Model saved ", saver.save(sess, Modeldir))
