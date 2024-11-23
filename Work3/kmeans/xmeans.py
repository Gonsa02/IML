import numpy as np

from kmeans.kmeans import KMeans

class XMeans:
    def __init__(self, k_max, max_iters, seed=0, verbose=False):
        self.k_max = k_max
        self.max_iters = max_iters
        self.seed = seed
        self.verbose = verbose

        self._centroids = None
        self._best_k = None
    
    def _assign_labels(self, X):
        distances = np.array([
            np.linalg.norm(X - centroid, axis=1)
            for centroid in self._centroids
        ])
        return np.argmin(distances, axis=0)

    
    def _split_centroid(self, parent_centroid, cluster_points, epsilon=0.5):
        random_direction = np.random.randn(parent_centroid.shape[0])
        random_direction /= np.linalg.norm(random_direction)

        cluster_diameter = np.max(np.linalg.norm(cluster_points - parent_centroid, axis=1))
        perturbation = epsilon * cluster_diameter

        child_1 = parent_centroid + perturbation * random_direction
        child_2 = parent_centroid - perturbation * random_direction

        return np.array([child_1, child_2])
    
    def _compute_bic(self, cluster_points_list, centroids):
        R = sum([len(cluster_points) for cluster_points in cluster_points_list])
        M = centroids[0].shape[0]
        K = len(centroids)

        variance = 0
        for idx, cluster_points in enumerate(cluster_points_list):
            variance += np.sum((cluster_points - centroids[idx])**2)

        # Same nb of centroids as points, we assume point = centroid so there is no variance
        if R - K == 0:
            variance = 0
        else:
            variance /= (R - K)

        # Just in case we have log(0)
        if variance == 0:
            variance = 1e-6

        log_likelihood = 0
        for idx, cluster_points in enumerate(cluster_points_list):
            R_n = len(cluster_points)
            log_likelihood += (
                - 0.5 * R_n * np.log(2 * np.pi)
                - 0.5 * R_n * M * np.log(variance)
                - 0.5 * (R_n - K)
                + R_n * np.log(R_n)
                - R_n * np.log(R)
            )

        penalty = 0.5 * (K-1 + M*K + 1) * np.log(R)
        return log_likelihood - penalty
    
    def fit_predict(self, X):
        np.random.seed(self.seed)

        X = np.array(X)

        self._centroids = [np.mean(X, axis=0)]
        labels = np.zeros(X.shape[0], dtype=int)

        k_current = 1
        
        still_dividing = True

        while still_dividing and k_current < self.k_max:
            division_ocurred = False
            new_centroids = []

            for cluster_id in range(k_current):
                #Take only subset of points belonging to that cluster
                cluster_points = X[labels == cluster_id]

                if len(cluster_points) == 1:
                    new_centroids.append(self._centroids[cluster_id])
                elif len(cluster_points) > 1:

                    child_centroids = self._split_centroid(self._centroids[cluster_id], cluster_points)

                    kmeans = KMeans(k=2, max_iters=self.max_iters, seed=self.seed, verbose=self.verbose)
                    child_labels = kmeans.fit_predict(cluster_points, initial_centroids=child_centroids)
                    centroids = kmeans.get_centroids()

                    bic_no_division = self._compute_bic([cluster_points], [self._centroids[cluster_id]])
                    bic_division = self._compute_bic(
                        [cluster_points[child_labels == 0], cluster_points[child_labels == 1]],
                        centroids
                    )
                
                    if bic_division > bic_no_division:
                        division_ocurred = True
                        new_centroids.extend(centroids)
                    else:
                        new_centroids.append(self._centroids[cluster_id])

            if not division_ocurred:
                still_dividing = False
            
            self._centroids = np.array(new_centroids)
            k_current = len(self._centroids)
            labels = self._assign_labels(X)

        self._best_k = k_current
        return labels
    
    def predict(self, X):
        X = np.array(X)
        return self._assign_labels(X)

    def get_best_k(self):
        return self._best_k