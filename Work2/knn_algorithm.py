import numpy as np
import pandas as pd

class KnnAlgorithm:
    # https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

    def __init__(self, X_train=None, Y_train = None):
        if X_train is not None:
            self.train(X_train, Y_train)

    def train(self, X_train, Y_train):
        self.original_X_train = X_train
        self.original_Y_train = Y_train

        self.X_train = X_train.to_numpy()
        self.Y_train = Y_train.to_numpy()

        self.label_to_integer = {label: idx for idx, label in enumerate(np.unique(self.Y_train))}
        self.integer_to_label = {idx: label for label, idx in self.label_to_integer.items()}
        self.num_classes = len(self.label_to_integer)

        self.Y_train = np.vectorize(self.label_to_integer.get)(self.Y_train) #Convert all classes to integers

    def _minkowski_distance(self, x1, x2, r=1):
        distance = 0.0
        for i in range(len(x1)):
            distance += (abs(x1[i] - x2[i])) ** r
        return distance ** (1 / r)

    def _cosine_distance(self, x1, x2):
        cosine_similarity = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return 1 - cosine_similarity

    def _majority_class_vote(self, neighborhoods_index):
        vote_counter = np.zeros(self.num_classes, dtype=int)

        for neighbor_index in neighborhoods_index:
            vote_counter[self.Y_train[neighbor_index]] += 1

        # In case of tie, we choose the class of the nearest neighbor
        return np.argmax(vote_counter)

    def _inverse_distance_weighted_vote(self, x, neighborhoods_index, p=1):
        vote_counter = np.zeros(self.num_classes)

        for neighbor_index in neighborhoods_index:
            distance = self._minkowski_distance(x, self.X_train[neighbor_index])
            weight = 1 / np.pow(distance, 2)
            vote_counter[self.Y_train[neighbor_index]] += weight

        return np.argmax(vote_counter)

    def _sheppards_work_vote(self, x, neighborhoods_index):
        vote_counter = np.zeros(self.num_classes)

        for neighbor_index in neighborhoods_index:
            distance = self._minkowski_distance(x, self.X_train[neighbor_index])
            weight = np.pow(np.e, -distance)
            vote_counter[self.Y_train[neighbor_index]] += weight

        return np.argmax(vote_counter)

    def _get_k_nearest_neighborhood(self, x, k):
        distances = np.zeros_like(self.Y_train)

        for i, x_train in enumerate(self.X_train):
            distances[i] = self._minkowski_distance(x, x_train)

        # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
        return np.argpartition(distances, k)[:k]


    def predict(self, X, k):
        X = X.to_numpy()
        predictions = np.zeros(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            neighborhoods_index = self._get_k_nearest_neighborhood(x, k)
            predictions[i] = self._majority_class_vote(neighborhoods_index)

        return pd.Series(np.vectorize(self.integer_to_label.get)(predictions), name=self.original_Y_train.name)