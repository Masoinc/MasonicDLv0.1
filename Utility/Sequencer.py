import numpy as np
import pandas as pd


# Datadir = "E:\PyCharmProjects\MasonicDLv0.1\Database\\"

def normal(data):
    data = (data - data.min()) / (data.max() - data.min())
    return data

    # data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))


def get_seq():
    data = pd.read_csv("E:\PyCharmProjects\MasonicDLv0.1\Database\MultiDB_headerless.csv", header=None,
                       skip_blank_lines=True)
    seq_size = 10
    x, y = [], []
    for i in range(0, 10, 2):
        qi, si = normal(data[i]), normal(data[i + 1])
        qi, si = qi.dropna(axis=0, how='all'), si.dropna(axis=0, how='all')
        for k in range(len(qi) - seq_size):
            # [10, 10]
            x.append(np.transpose([qi[k:k + seq_size], si[k:k + seq_size]]))
            y.append(np.expand_dims(qi[k + seq_size], axis=0))
    return x, y


def get_raw():
    data = pd.read_csv("E:\PyCharmProjects\MasonicDLv0.1\Database\MultiDB_noheader.csv", header=None)
    datas = []
    for i in range(0, data.shape[1], 2):
        q = normal(data[i])
        s = normal(data[i + 1])
        datas.append(np.transpose([q.dropna(axis=0, how='all'), s.dropna(axis=0, how='all')]))
    return datas


if __name__ == '__main__':
    get_raw()
    # for i in range(0,  )
# 单部电视剧[time, 2]
# [2, 10, 794]
# x = pd.DataFrame(x)
# x.to_pickle(Datadir + "MultiDB_X.pkl")
# x.to_csv(Datadir + "MultiDB_X.csv")
# y = pd.DataFrame(y)
# y.to_pickle(Datadir + "MultiDB_Y.pkl")
# y.to_csv(Datadir + "MultiDB_Y.csv")
# final x shape: [[10, 10], [10, 10], ... [10, 10]]

# x = pd.read_csv("E:\PyCharmProjects\MasonicDLv0.1\Database\MultiDB_X.csv", header=None)
# y = pd.read_pickle(Datadir + "MultiDB_Y.pkl")
# x=pd.read_pickle(Datadir + "MultiDB_X.pkl")
# x.info()
# print(x[1])
