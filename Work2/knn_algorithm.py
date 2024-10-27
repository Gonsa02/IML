import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2

class KnnAlgorithm:
    # https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

    def __init__(self, X_train=None, y_train=None):

        self.distance_metrics = {
            "minkowski_r1": lambda x1, x2: self._minkowski_distance(x1, x2, r=1),
            "minkowski_r2": lambda x1, x2: self._minkowski_distance(x1, x2, r=2),
            "hamming": self._hamming_distance
        }

        self.voting_policies = {
            "majority": lambda x, index, metric: self._majority_class_vote(index),
            "idw": lambda x, index, metric: self._inverse_distance_weighted_vote(x, index, metric),
            "sheppard": lambda x, index, metric: self._sheppards_work_vote(x, index, metric),
        }

        self.weighting_methods = {
            "eq_weight": lambda X, y: np.ones(X.shape[1]),
            "information_gain": lambda X, y: mutual_info_classif(X, y),
            "chi2": lambda X, y: np.nan_to_num(chi2(X, y)[0], nan=0.0),
        }

        if X_train is not None:
            self.train(X_train, y_train)

    def train(self, X_train, y_train, weight_method="eq_weight"):
        self.X_train = X_train.to_numpy()
        self.y_train = y_train.to_numpy()

        self.label_to_integer = {label: idx for idx, label in enumerate(np.unique(self.y_train))}
        self.integer_to_label = {idx: label for label, idx in self.label_to_integer.items()}
        self.num_classes = len(self.label_to_integer)

        self.y_train = np.vectorize(self.label_to_integer.get)(self.y_train) #Convert all classes to integers

        self.feature_weights = self.weighting_methods[weight_method](self.X_train, self.y_train)
        self.X_train *= self.feature_weights

        self.y_train_name = y_train.name

    def _minkowski_distance(self, x1, x2, r=1):
        distance = 0.0
        for i in range(len(x1)):
            distance += (abs(x1[i] - x2[i])) ** r
        return distance ** (1 / r)
    
    def _hamming_distance(self, x1, x2):
        distance = 0
        for i in range(len(x1)):
            if x1[i] != x2[i]:
                distance += 1
                
        return distance

    def _majority_class_vote(self, neighborhoods_index):
        vote_counter = np.zeros(self.num_classes, dtype=int)

        for neighbor_index in neighborhoods_index:
            vote_counter[self.y_train[neighbor_index]] += 1

        # In case of tie, we choose the class of the nearest neighbor
        return np.argmax(vote_counter)

    def _inverse_distance_weighted_vote(self, x, neighborhoods_index, d_metric, p=1):
        vote_counter = np.zeros(self.num_classes)

        for neighbor_index in neighborhoods_index:
            distance = self.distance_metrics[d_metric](x, self.X_train[neighbor_index])
            if distance == 0:
                weight = 1
            else:
                weight = 1 / np.pow(distance, p)
            vote_counter[self.y_train[neighbor_index]] += weight

        return np.argmax(vote_counter)

    def _sheppards_work_vote(self, x, neighborhoods_index, d_metric):
        vote_counter = np.zeros(self.num_classes)

        for neighbor_index in neighborhoods_index:
            distance = self.distance_metrics[d_metric](x, self.X_train[neighbor_index])
            weight = np.pow(np.e, -distance)
            vote_counter[self.y_train[neighbor_index]] += weight

        return np.argmax(vote_counter)

    def _get_k_nearest_neighborhood(self, x, k, d_metric):
        distances = np.zeros_like(self.y_train)

        for i, x_train in enumerate(self.X_train):
            distances[i] = self.distance_metrics[d_metric](x, x_train)

        # https://stackoverflow.com/questions/34226400/find-the-index-of-the-k-smallest-values-of-a-numpy-array
        return np.argpartition(distances, k)[:k]


    def predict(self, X, k, d_metric, v_policy):
        X = X.to_numpy()
        X *= self.feature_weights

        predictions = np.zeros(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            neighborhoods_index = self._get_k_nearest_neighborhood(x, k, d_metric)
            predictions[i] = self.voting_policies[v_policy](x, neighborhoods_index, d_metric)

        return pd.Series(np.vectorize(self.integer_to_label.get)(predictions), name=self.y_train_name)