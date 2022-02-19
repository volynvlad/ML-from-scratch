from collections import Counter

import numpy as np

from utils import minkowsi_distane


class KNN:
    def __init__(self, k=3, p=2) -> None:
        self.k = k
        self.p = p

    def fit(self, X, y):
        # store data and labels
        self.X_train = X
        self.y_train = y

    def _predict(self, x):
        # compute distances
        distance = [minkowsi_distane(x, x_train, self.p) for x_train in self.X_train]

        # get k nearest samples, labels
        k_indices = np.argsort(distance)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote, most common class label
        counter = Counter(k_nearest_labels)
        most_common = counter.most_common(1)

        return most_common[0][0]

    def predict(self, X):
        predicted_lables = [self._predict(x) for x in X]
        return np.array(predicted_lables)
