import numpy as np
from .kmeans import KMeans


class fast_global_k_means(KMeans):
    def __init__(self, k=2, max_iters=100, distance='euclidean', seed=0, verbose=False):
        self.k = k
        self.max_iters = max_iters
        self.distance = distance
        self.seed = seed
        self.verbose = verbose

    # X (n, d), centroids (k, d)
    def _compute_bounds(self, X, centroids):
        distances = np.array([np.linalg.norm(X - c, axis=1)
                             for c in centroids])
        # distances from x to their nearest cluster
        min_distances = np.min(distances, axis=0)
        # reduction in error obtained for each sample (as a new cluster)
        bounds = [np.sum(np.maximum(
            min_distances - np.linalg.norm(X - xn, axis=1)**2, 0)) for xn in X]
        return np.array(bounds)

    def fit_predict(self, X):
        X = np.array(X)
        n = X.shape[0]

        first_centroid = np.mean(X, axis=0).reshape(
            1, -1)  # converts 2D (1, d)
        self.centroids = first_centroid

        for k_value in range(2, self.k+1):
            if self.verbose:
                print("Adding cluster", k_value)

            bounds = self._compute_bounds(X, self.centroids)
            new_centroid_idx = np.argmax(bounds)
            new_centroid = X[new_centroid_idx].reshape(1, -1)

            init_centroids = np.vstack([self.centroids, new_centroid])

            km = KMeans(
                k=k_value, distance=self.distance, seed=self.seed)
            _ = km.fit_predict(X, initial_centroids=init_centroids)

            self.centroids = km.get_centroids()

        final_kmeans = KMeans(
            k=self.k, distance=self.distance, seed=self.seed)
        labels = final_kmeans.fit_predict(
            X, initial_centroids=self.centroids)

        self.n_iter_ = final_kmeans.get_iterations()

        return labels

    def get_iterations(self):
        return self.n_iter_
