import numpy as np


class Distances:
    @staticmethod
    def _minkowski_distance(X, centroid, r=1):
        return np.sum(np.abs(X-centroid)**r, axis=1)**(1/r)

    @staticmethod
    def manhattan(X, centroid):
        return Distances._minkowski_distance(X, centroid, r=1)

    @staticmethod
    def euclidean(X, centroid):
        return Distances._minkowski_distance(X, centroid, r=2)

    @staticmethod
    def cosine_distance(X, centroid):
        similarity = np.dot(X, centroid) / (np.linalg.norm(X,
                                                           axis=1) * np.linalg.norm(centroid))
        return 1-similarity


class KMeans:
    def __init__(self, k=2, max_iters=30, distance='Euclidean', seed=0):
        self.k = k
        self.max_iters = max_iters
        self.distance = KMeans._get_distance(distance)
        self.seed = seed

        self.centroids_idxs = []

    @staticmethod
    def _get_distance(distance):
        if distance == 'Manhattan':
            return Distances.manhattan
        elif distance == 'Euclidean':
            return Distances.euclidean
        else:
            return Distances.cosine_distance

    def _assign_samples(X):
        return

    def _update_centroids():
        return

    def fit(self, X):
        np.random.seed(self.seed)

        # Centroid initialization
        self.centroids_idxs = np.random.choice(X.shape[0], self.k)
