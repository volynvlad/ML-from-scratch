from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

from algorithms.knn import KNN


def test_knn():

    # cmap = ListedColormap(["red", "green", "blue"])

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=300)

    # print(X_train.shape)
    # print(y_train.shape)

    # print(X_train[0])
    # print(y_train[0])

    # plt.figure()
    # plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolors='k', s=20)
    # plt.show()

    ks = [3, 5, 7]
    ps = [1, 2, 3]
    for k in ks:
        for p in ps:
            clf_impl = KNN(k=k, p=p)
            clf = KNeighborsClassifier(n_neighbors=k, p=p, algorithm="brute")

            clf_impl.fit(X_train, y_train)
            clf.fit(X_train, y_train)

            for pred_impl, pred in zip(clf_impl.predict(X_test), clf.predict(X_test)):
                is_equal = True
                msg = ""
                if pred_impl != pred:
                    is_equal = False
                    msg = f"Not equal, {pred_impl=}, {pred=}, {p=}, {k=}"
                assert is_equal, msg
