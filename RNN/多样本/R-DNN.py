import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

Datadir = "G:\Idea\MasonicDLv0.1\Database\MultiDBpartI_noheader.csv"
TensorBoarddir = "G:\Idea\MasonicDLv0.1\Board\Multi-Sample-R-DNN"
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt_title = "R-DNN Examine on Sample《继承人》"

with tf.name_scope(name='Hyperparameter'):
    step = 2001
    seq_size = 5
    vector_size = 2
    batch_size = 20
    test_size = 20

    train_percent = 0.9
    hidden_layer_size = 5

    reg = True
    reg_rate = 0.0015

    global_step = tf.Variable(0, name="global_step")
    learning_rate = tf.train.exponential_decay(
        learning_rate=0.1,
        global_step=global_step,
        decay_steps=100,
        decay_rate=0.9,
        staircase=True)

with tf.name_scope(name='Placeholder'):
    X = tf.placeholder("float", [None, seq_size, vector_size])
    Y = tf.placeholder("float", [None, 1])

    W = {
        "w1": tf.Variable(tf.random_normal([hidden_layer_size, 15])),
        "w2": tf.Variable(tf.random_normal([15, 8])),
        "w3": tf.Variable(tf.random_normal([8, 1])),
        "w4": tf.Variable(tf.random_normal([5, 1])),
        "b1": tf.Variable(tf.random_normal([1])),
        "b2": tf.Variable(tf.random_normal([1])),
        "b3": tf.Variable(tf.random_normal([1])),
        "b4": tf.Variable(tf.random_normal([1]))
    }

with tf.name_scope(name='DataProcessing'):
    def normal(data):
        data = (data - data.min()) / (data.max() - data.min())
        return data


    def get_seq():
        data = pd.read_csv(Datadir, header=None, skip_blank_lines=True)
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
    X = tf.transpose(X, [1, 0, 2])
    X = tf.reshape(X, [-1, vector_size])
    X = tf.split(X, seq_size, 0)

    cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_layer_size, use_peepholes=True)
    outputs, _ = tf.nn.static_rnn(cell, X, dtype=tf.float32)
    y_ = tf.nn.tanh(tf.matmul(outputs[-1], W["w1"]) + W["b1"])
    y_ = tf.nn.tanh(tf.matmul(y_, W["w2"]) + W["b2"])
    y_ = tf.nn.tanh(tf.matmul(y_, W["w3"]) + W["b3"])

    W_ = [W['w1'], W['w2'], W['w3'], W['w4']]

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
    # tf.summary.scalar('loss', loss)
tf.summary.scalar('train_loss', loss)
train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

merged = tf.summary.merge_all()
with tf.Session() as sess:
    time1 = time.time()
    train_writer = tf.summary.FileWriter(TensorBoarddir, sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(step):
        train_loss, test_loss = 0, 0
        loss_sum = 0
        k = 0
        for end in range(batch_size, len(trY), batch_size):
            begin = end - batch_size
            sampleX = trX[begin:end]
            sampleY = trY[begin:end]
            summary, train_loss, _ = sess.run([merged, loss, train_op], feed_dict={X: sampleX, Y: sampleY})
            if k == 0:
                train_writer.add_summary(summary, i)
                k = 1
            loss_sum += train_loss
        if i % 100 == 0 and i > 1:
            time2 = time.time()
            print('Time used: {0}s'.format(time2 - time1))
            test_indices = np.arange(len(teX))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
            trX_sliced, trY_sliced = [], []
            for t in test_indices:
                trX_sliced.append(teX[t])
                trY_sliced.append(teY[t])
            tf.summary.scalar('test_loss', test_loss)
            summary, test_loss = sess.run([merged, loss], feed_dict={X: trX_sliced, Y: trY_sliced})

            train_writer.add_summary(summary, i)
            print("Train Step: ", i)
            print("Accuracy on train/test: %f/%f" % (
                pow(loss_sum / test_size, 0.5), pow(test_loss, 0.5)))
            print('Time used: {0}s'.format(time2 - time1))

            if i == step - 1:
                time2 = time.time()
                print('Time used: {0}s'.format(time2 - time1))

                preY = sess.run(y_, feed_dict={X: trX_sliced})
                realY = np.array(trY_sliced)
                realY = np.squeeze(realY)
                x_axis = np.arange(0, np.shape(realY)[0])

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

                sample = pd.read_csv("G:\Idea\MasonicDLv0.1\Database\Sample_JiChengRen.csv", header=None,
                                     skip_blank_lines=True)

                seq1, seq2 = normal(sample[1]), normal(sample[3])
                seq1, seq2 = seq1.dropna(axis=0, how='all'), seq2.dropna(axis=0, how='all')
                sampleX, sampleY = [], []
                for k in range(len(seq1) - seq_size):
                    # [10, 10]
                    sampleX.append(np.transpose([seq1[k:k + seq_size], seq2[k:k + seq_size]]))
                    sampleY.append(np.expand_dims(seq1[k + seq_size], axis=0))

                test_loss = sess.run(loss, feed_dict={X: sampleX, Y: sampleY})

                preY = sess.run(y_, feed_dict={X: sampleX})
                preY = np.squeeze(preY)

                realY = np.squeeze(np.array(sampleY))
                x_axis = np.arange(0, np.shape(realY)[0])

                print("Accuracy on SAMPLE: {0}".format(pow(test_loss, 0.5)))

                plt.figure(figsize=(16, 10))
                plt.plot(x_axis, preY, label="Prediction")
                plt.plot(x_axis, realY, label="Observation")
                plt.legend()
                plt.title(plt_title)
                plt.show()
