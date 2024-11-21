import numpy as np
from collections import Counter


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

    def __init__(self, k=2, max_iters=30, distance='euclidean', seed=0, verbose=False):
        self.k = k
        self.max_iters = max_iters
        self.distance = KMeans._get_distance(distance)
        self.seed = seed
        self.verbose = verbose

        self.centroids = None

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

    # change tolerance
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

    def fit_predict(self, X):
        np.random.seed(self.seed)

        X = np.array(X)

        # Centroid initialization
        centroids_idx = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[centroids_idx]

        for i in range(self.max_iters):
            if self.verbose:
                print(f"Iteration: {i}/{self.max_iters}")

            labels = self._assign_samples(X)

            # Update centroids based on current assignments
            new_centroids = self._update_centroids(X, labels)
            if self._check_convergence(new_centroids):
                if (self.verbose):
                    print(f"Converged at iteration {i+1}")
                break

            self.centroids = new_centroids
            if (i == self.max_iters-1):
                print(f"No convergence")

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

    @staticmethod
    def compute_accuracy(predicted_labels, true_labels):
        predicted_labels = np.array(predicted_labels)
        true_labels = np.array(true_labels)

        label_mapping = {}
        clusters = np.unique(predicted_labels)

        for c in clusters:
            samples_idxs = np.where(predicted_labels == c)

            true_labels_cluster = true_labels[samples_idxs]
            if len(true_labels_cluster) > 0:
                label_mapping[c] = Counter(
                    true_labels_cluster).most_common(1)[0][0]
            else:
                label_mapping[c] = -1

        matched_predicted_labels = np.array(
            [label_mapping[c] for c in predicted_labels])

        accuracy = np.mean(matched_predicted_labels == true_labels)
        return accuracy
