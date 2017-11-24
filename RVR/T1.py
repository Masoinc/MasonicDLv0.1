from sklearn.utils.estimator_checks import check_estimator
from skbayes.rvm_ard_models import RegressionARD, ClassificationARD, RVR, RVC
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error

# parameters
n = 5000

# generate data set
np.random.seed(0)
Xc = np.ones([n, 1])
Xc[:, 0] = np.linspace(-5, 5, n)
Yc = 10 * np.sinc(Xc[:, 0]) + np.random.normal(0, 1, n)
X, x, Y, y = train_test_split(Xc, Yc, test_size=0.5, random_state=0)

# train rvr
rvm = RVR(gamma=1, kernel='rbf')
t1 = time.time()
rvm.fit(X, Y)
t2 = time.time()
y_hat, var = rvm.predict_dist(x)
rvm_err = mean_squared_error(y_hat, y)
rvs = np.sum(rvm.active_)
print("RVM error on test set is {0}, number of relevant vectors is {1}, time {2}".format(rvm_err, rvs, t2 - t1))

# train svr
svr = GridSearchCV(SVR(kernel='rbf', gamma=1), param_grid={'C': [0.001, 0.1, 1, 10, 100]}, cv=10)
t1 = time.time()
svr.fit(X, Y)
t2 = time.time()
svm_err = mean_squared_error(svr.predict(x), y)
svs = svr.best_estimator_.support_vectors_.shape[0]
print(print()
      )

# # plot test vs predicted data
# plt.figure(figsize=(16, 10))
# plt.plot(x[:, 0], y, "b+", markersize=3, label="test data")
# plt.plot(x[:, 0], y_hat, "rD", markersize=3, label="mean of predictive distribution")
#
# # plot one standard deviation bounds
# plt.plot(x[:, 0], y_hat + np.sqrt(var), "co", markersize=3, label="y_hat +- std")
# plt.plot(x[:, 0], y_hat - np.sqrt(var), "co", markersize=3)
# plt.plot(rvm.relevant_vectors_, Y[rvm.active_], "co", markersize=12, label="relevant vectors")
# plt.legend()
# plt.title("RVM")
# plt.show()
