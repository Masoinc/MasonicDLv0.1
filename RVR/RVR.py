from skrvm import RVR
from skrvm import RVC
from sklearn.datasets import load_iris

X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = RVR(kernel='linear')
# clf = RVR(kernel='rbf')
# clf = RVR(kernel='poly')
clf.fit(X, y)

RVR(alpha=1e-06,
    beta=1e-06,
    beta_fixed=False,
    bias_used=True,
    coef0=0.0,
    coef1=None,
    degree=3,
    kernel='linear',
    n_iter=3000,
    threshold_alpha=1000000000.0,
    tol=0.001,
    verbose=True)

print(clf.predict([[1, 1]]))

# clf = RVC()
# clf.fit(load_iris().data, load_iris().target)
# RVC(alpha=1e-06, beta=1e-06, beta_fixed=False, bias_used=True, coef0=0.0,
#     coef1=None, degree=3, kernel='rbf', n_iter=3000, n_iter_posterior=50,
#     threshold_alpha=1000000000.0, tol=0.001, verbose=False)
# score = clf.score(load_iris().data, load_iris().target)
# print(score)
