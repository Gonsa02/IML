import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment


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

    def __init__(self, k=2, max_iters=100, distance='euclidean', seed=0, verbose=False):
        self.k = k
        self.max_iters = max_iters
        self.distance = KMeans._get_distance(distance)
        self.seed = seed
        self.verbose = verbose

        self.centroids = None
        self.n_iter_ = None

    @staticmethod
    def _get_distance(distance):
        if distance == 'manhattan':
            return Distances.manhattan
        elif distance == 'euclidean':
            return Distances.euclidean
        elif distance == 'cosine':
            return Distances.cosine_distance
        else:
            raise ValueError(
                "Unsupported distance metric. Choose 'euclidean', 'manhattan', or 'cosine'.")

    def _assign_samples(self, X):
        distances = np.array([self.distance(X, centroid)
                             for centroid in self.centroids])
        return np.argmin(distances, axis=0)

    def _check_convergence(self, new_centroids):
        return np.allclose(self.centroids, new_centroids, atol=1e-5)

    def _update_centroids(self, X, labels):
        new_centroids = np.zeros_like(self.centroids)
        for i in range(self.k):
            cluster_samples = X[labels == i]
            if len(cluster_samples) > 0:
                new_centroids[i] = cluster_samples.mean(axis=0)
            else:
                # In case a cluster does not have any sample, reassign
                new_centroids[i] = X[np.random.randint(0, X.shape[0])]
        return new_centroids

    def fit_predict(self, X, initial_centroids=None):
        np.random.seed(self.seed)

        X = np.array(X)

        # Centroid initialization
        if initial_centroids is None:
            centroids_idx = np.random.choice(X.shape[0], self.k, replace=False)
            self.centroids = X[centroids_idx]
        else:
            self.centroids = initial_centroids

        for i in range(self.max_iters):
            if self.verbose:
                print(f"Iteration: {i}/{self.max_iters}")

            labels = self._assign_samples(X)

            # Update centroids based on current assignments
            new_centroids = self._update_centroids(X, labels)
            if self._check_convergence(new_centroids):
                self.n_iter_ = i
                if (self.verbose):
                    print(f"Converged at iteration {i+1}")
                break

            self.centroids = new_centroids
            if (i == self.max_iters-1):
                self.n_iter_ = self.max_iters
                if (self.verbose):
                    print(f"No convergence")

        labels = self._assign_samples(X)
        return labels

    def predict(self, X):
        if self.centroids is None:
            raise Exception("Model has not been fitted yet.")

        X = np.array(X)
        return self._assign_samples(X)

    def get_centroids(self):
        if self.centroids is None:
            raise Exception("Model has not been fitted yet.")

        return self.centroids

    def get_iterations(self):
        if self.n_iter_ is None:
            raise Exception("Model has not been fitted yet.")

        return self.n_iter_
