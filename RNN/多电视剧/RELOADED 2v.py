import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Datadir = "E:\PyCharmProjects\MasonicDLv0.1\Database\MultiDB_noheader.csv"

with tf.name_scope(name='Hyperparameter'):
    step = 2000

    seq_size = 10
    vector_size = 2
    batch_size = 50
    test_size = 30
    hidden_layer_size = 10

    global_step = tf.Variable(0, name="global_step")
    learning_rate = tf.train.exponential_decay(
        learning_rate=0.001,
        global_step=global_step,
        decay_steps=30,
        decay_rate=0.9,
        staircase=True)

    train_percent = 0.7
with tf.name_scope(name='Placeholder'):
    X = tf.placeholder("float", [None, seq_size, vector_size])
    Y = tf.placeholder("float", [None, seq_size])

    W = {
        "w1": tf.Variable(tf.random_normal([hidden_layer_size, 50])),
        "w2": tf.Variable(tf.random_normal([50, 25])),
        "w3": tf.Variable(tf.random_normal([25, 10])),
        "w4": tf.Variable(tf.random_normal([10, 1])),
        "b1": tf.Variable(tf.random_normal([1])),
        "b2": tf.Variable(tf.random_normal([1])),
        "b3": tf.Variable(tf.random_normal([1])),
        "b4": tf.Variable(tf.random_normal([1]))
    }

with tf.name_scope(name='DataProcessing'):
    def normal(data):
        data = (data - data.min()) / (data.max() - data.min())
        return data


    def get_multi():
        data = pd.read_csv(Datadir, header=None, skip_blank_lines=True)

        x, y = [], []
        for i in range(0, 10, 2):
            qi, si = normal(data[i]), normal(data[i + 1])
            qi, si = qi.dropna(axis=0, how='all'), si.dropna(axis=0, how='all')
            for k in range(len(qi) - seq_size):
                # [10, 10]
                x.append(np.transpose([qi[k:k + seq_size], si[k:k + seq_size]]))
                # y.append(np.expand_dims(qi[k + seq_size], axis=0))
                y.append(qi[k + 1:k + seq_size + 1])
        return x, y


    x, y = get_multi()
    tr_num = int(train_percent * len(y))

    trX, trY = x[:tr_num], y[:tr_num]
    teX, teY = x[tr_num:], y[tr_num:]

with tf.name_scope(name='NeuralNetwork'):
    def nn(X, W, seq_size, vector_size):
        X = tf.transpose(X, [1, 0, 2])
        X = tf.reshape(X, [-1, vector_size])
        X = tf.split(X, seq_size, 0)
        cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_layer_size)

        outputs, _ = tf.nn.static_rnn(cell, X, dtype=tf.float32)

        # mcell = []
        # for layer in range(2):
        #     cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_layer_size)
        #     # 单层LSTM
        #     cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=1.0)
        #     # drop-out层
        #     mcell.append(cell)
        # mcell = tf.nn.rnn_cell.MultiRNNCell(cells=mcell, state_is_tuple=True)

        outputs, _ = tf.nn.static_rnn(cell, X, dtype=tf.float32)

        w1, w2, w3, w4 = W['w1'], W['w2'], W['w3'], W['w4']

        w1 = tf.tile(input=tf.expand_dims(w1, 0), multiples=[tf.shape(X)[0], 1, 1])
        w2 = tf.tile(input=tf.expand_dims(w2, 0), multiples=[tf.shape(X)[0], 1, 1])
        w3 = tf.tile(input=tf.expand_dims(w3, 0), multiples=[tf.shape(X)[0], 1, 1])
        w4 = tf.tile(input=tf.expand_dims(w4, 0), multiples=[tf.shape(X)[0], 1, 1])

        fc1 = tf.nn.tanh(tf.matmul(outputs, w1) + W["b1"])
        fc2 = tf.nn.tanh(tf.matmul(fc1, w2) + W["b2"])
        fc3 = tf.nn.tanh(tf.matmul(fc2, w3) + W["b3"])
        fc4 = tf.nn.tanh(tf.matmul(fc3, w4) + W["b4"])
        trans = tf.squeeze(fc4)
        y_ = tf.transpose(trans)
        return y_

with tf.name_scope(name='TrainSettings'):
    y_ = nn(X, W, seq_size, vector_size)
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
        if i % 10 == 0:
            test_indices = np.arange(len(teX))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
            trX_sliced, trY_sliced = [], []
            for t in test_indices:
                trX_sliced.append(trX[t])
                trY_sliced.append(trY[t])
            test_loss = sess.run(loss, feed_dict={X: trX_sliced, Y: trY_sliced})
            print("Train Step: ", i)
            print("RMSE on train/test: {0}/{1}".format(pow((loss_sum / batch_size), 0.5), pow(test_loss, 0.5)))
            if i % 500 == 0 and i > 1:
                preY = sess.run(y_, feed_dict={X: trX_sliced})
                preY = preY[:, -1]
                realY = np.array(trY_sliced)
                realY = realY[:, -1]

                x_axis = np.arange(0, test_size)

                plt.figure(figsize=(16, 10))
                ax = plt.gca()
                ax.xaxis.grid(True)
                ax.set_xticks(x_axis)
                # plt.vlines(data_size - test_size, 0, 1, colors="c", linestyles="dashed", label='train/test split')
                plt.scatter(x_axis, preY, label="Prediction")
                plt.scatter(x_axis, realY, label="Observation")
                plt.legend()
                plt.title("RNN")
                plt.show()
