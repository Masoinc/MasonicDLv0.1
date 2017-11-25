import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

Datadir = "E:\PyCharmProjects\MasonicDLv0.1\Database\楚乔传_爱奇艺指数.csv"

with tf.name_scope(name='Hyperparameter'):
    data_col = 1

    early_stopping_rate = 0.001
    train_step = 8000
    global_step = tf.Variable(0, name="global_step")
    learning_rate = tf.train.exponential_decay(
        learning_rate=0.001,
        global_step=global_step,
        decay_steps=100,
        decay_rate=0.9,
        staircase=True)

    reg = True
    reg_rate = 0.003

    hidden_layer_size = 15

    seq_size = 10
    test_size = 0.2

with tf.name_scope(name='Placeholder'):
    X = tf.placeholder(tf.float32, [None, seq_size, 1])
    Y = tf.placeholder(tf.float32, [None, seq_size])

    W = {
        'w1': tf.Variable(tf.random_normal([hidden_layer_size, 30])),
        'w2': tf.Variable(tf.random_normal([30, 15])),
        'w3': tf.Variable(tf.random_normal([15, 1])),
        'b1': tf.Variable(tf.random_normal([1])),
        'b2': tf.Variable(tf.random_normal([1])),
        'b3': tf.Variable(tf.random_normal([1]))
    }

with tf.name_scope(name='DataProcessing'):
    def normal(data):
        data = (data - data.min()) / (data.max() - data.min())
        return data


    data = pd.read_csv(Datadir, header=None)
    data = normal(np.array(data[data_col]))
    data_size = np.shape(data)[0]

    seq, pre = [], []
    for i in range(data_size - seq_size + 1 - 1):
        seq.append(np.expand_dims(data[i: i + seq_size], axis=1))
        pre.append(data[i + 1:i + seq_size + 1])

    data_size = data_size - seq_size + 1 - 1
    test_size = int(data_size * test_size)
    trX = seq[:data_size - test_size]
    trY = pre[:data_size - test_size]
    teX = seq[data_size - test_size:]
    teY = pre[data_size - test_size:]
    realY = data[-test_size:]
with tf.name_scope(name='NeuralNetwork'):
    def rnn(X, W):
        w1, w2, w3 = W['w1'], W['w2'], W['w3']
        b1, b2, b3 = W['b1'], W['b2'], W['b3']
        W_ = [w1, w2, w3]
        cell = tf.nn.rnn_cell.GRUCell(hidden_layer_size)
        outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        w1 = tf.tile(input=tf.expand_dims(w1, 0), multiples=[tf.shape(X)[0], 1, 1])
        w2 = tf.tile(input=tf.expand_dims(w2, 0), multiples=[tf.shape(X)[0], 1, 1])
        w3 = tf.tile(input=tf.expand_dims(w3, 0), multiples=[tf.shape(X)[0], 1, 1])
        fc1 = tf.nn.tanh(tf.matmul(outputs, w1) + b1)
        fc2 = tf.nn.tanh(tf.matmul(fc1, w2) + b2)
        fc3 = tf.nn.tanh(tf.matmul(fc2, w3) + b3)
        y_ = tf.squeeze(fc3)
        return y_, W_

with tf.name_scope(name='TrainSettings'):
    y_, W_ = rnn(X, W)

    if reg:
        loss = tf.reduce_mean(tf.square(Y - y_)) + \
               tf.contrib.layers.l2_regularizer(reg_rate)(W_[0]) + \
               tf.contrib.layers.l2_regularizer(reg_rate)(W_[1]) + \
               tf.contrib.layers.l2_regularizer(reg_rate)(W_[2])
    else:
        loss = tf.reduce_mean(tf.square(Y - y_))

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_loss = 0
    nice = False
    for step in range(train_step):
        _, train_loss = sess.run([train_op, loss], feed_dict={X: trX, Y: trY})
        if step % 100 == 0 and prev_loss != -1:
            delta = 1 if prev_loss == 0 else (abs(train_loss - prev_loss) / prev_loss)
            prev_loss = train_loss

            prev_seq = teX[0]
            predict = []
            for i in range(test_size):
                next_seq = sess.run(y_, feed_dict={X: [prev_seq]})
                predict.append(next_seq[-1])
                prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))
            test_loss = mean_squared_error(predict, realY)
            print("Train Step={0}".format(step))
            print("Train RMSE={0}".format(pow(train_loss, 0.5)))
            print("Test RMSE={0}".format(pow(test_loss, 0.5)))

            if delta < early_stopping_rate:
                prev_loss = -1
                break
            if train_loss + test_loss < 0.05:
                nice = True
                print("Nice result for train={0}, test={1}".format(train_loss, test_loss))
                pre_train = sess.run(y_, feed_dict={X: trX})[:, -1]

                prev_seq = teX[0]
                predict = []
                for i in range(test_size):
                    next_seq = sess.run(y_, feed_dict={X: [prev_seq]})
                    predict.append(next_seq[-1])
                    prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

                pre = np.append(pre_train, np.array(predict))
                # pre = pre_train.append(np.array(predict))

                real_train = np.array(trY)[:, -1]

                real = np.append(real_train, realY)
                x_axis = np.arange(0, np.shape(real)[0])

                plt.figure(figsize=(16, 10))
                plt.vlines(data_size - test_size, 0, 1, colors="c", linestyles="dashed", label='train/test split')
                plt.plot(x_axis, pre, label="Prediction")
                plt.plot(x_axis, real, label="Observation")
                plt.legend()
                plt.title("RNN")
                plt.show()

        elif step == train_step - 1:
            pre_train = sess.run(y_, feed_dict={X: trX})[:, -1]

            prev_seq = teX[0]
            predict = []
            for i in range(test_size):
                next_seq = sess.run(y_, feed_dict={X: [prev_seq]})
                predict.append(next_seq[-1])
                prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

            pre = np.append(pre_train, np.array(predict))
            # pre = pre_train.append(np.array(predict))

            real_train = np.array(trY)[:, -1]

            real = np.append(real_train, realY)
            x_axis = np.arange(0, np.shape(real)[0])

            plt.figure(figsize=(16, 10))
            plt.vlines(data_size - test_size, 0, 1, colors="c", linestyles="dashed", label='train/test split')
            plt.plot(x_axis, pre, label="Prediction")
            plt.plot(x_axis, real, label="Observation")
            plt.legend()
            plt.title(u"GRU Prediction for 《楚乔传》")
            plt.show()
        elif nice:
            break
