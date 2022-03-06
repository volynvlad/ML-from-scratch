import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import SGDRegressor as sklearn_lr


from algorithms.linear_regression import LinearRegression as implemented_lr

import sklearn


X, y = datasets.make_regression(n_samples=300,
                                n_features=2,
                                noise=20,
                                random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=52)


def plot_figure(X, y):
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], y, color='b', marker='o', s=30)
    plt.show()


# plot_figure(X, y)


def test_lr():

    n_iter = 3000
    lr = 0.003
    clf_impl = implemented_lr(n_iters=n_iter, lr=lr)
    clf = sklearn_lr(max_iter=n_iter,
                     alpha=0,
                    #  epsilon=lr,
                     eta0=lr,
                     shuffle=False,
                     learning_rate="constant",
                     n_iter_no_change=n_iter,
                     loss="squared_error",
                     tol=None)

    clf_impl.fit(X_train, y_train)
    clf.fit(X_train, y_train)

    print(clf_impl.weights, clf_impl.bias)
    print(clf.coef_, clf.intercept_[0])

    assert clf_impl.weights == clf.coef_, f"Weights are not equal: {clf_impl.weights=}, {clf.coef_=}"
    assert clf_impl.bias == clf.intercept_[0], f"Weights are not equal: {clf_impl.bias=}, {clf.intercept_[0]=}"
