import numpy as np
import pandas as pd

class GSFuzzyCMeans:
    def __init__(self, n_clusters=3, m=2, xi=0.5, max_iter=100, error=1e-5, random_state=None):
        """
        Initialize the GSFuzzyCMeans clustering with the gsξ-FCM suppression scheme.

        Parameters:
        - n_clusters: Number of clusters.
        - m: Fuzziness parameter (m > 1).
        - xi: Suppression parameter ξ (0 ≤ xi ≤ 1).
        - max_iter: Maximum number of iterations.
        - error: Convergence threshold.
        - random_state: Seed for random number generation.
        """
        self.n_clusters = n_clusters
        self.m = m
        self.xi = xi
        self.max_iter = max_iter
        self.error = error
        self.random_state = random_state
        self.n_iter_ = 0  # Initialize the iteration counter

    def fit(self, X):
        """
        Fit the model to the data X.

        Parameters:
        - X: Data array of shape (n_samples, n_features).
        """
        X = self._check_array(X)
        self.X = X
        self.n_samples, self.n_features = X.shape

        self._initialize_U()

        for iteration in range(self.max_iter):
            U_old = self.U.copy()

            # Calculate cluster centers
            self.centers = self._calculate_cluster_centers()

            # Update membership matrix with suppression
            self.U = self._update_membership_matrix()

            # Increment the iteration counter
            self.n_iter_ = iteration + 1

            # Check for convergence
            if np.linalg.norm(self.U - U_old) < self.error:
                # print(f"Converged at iteration {self.n_iter_}")
                break
        else:
            print(f"Reached maximum iterations ({self.max_iter}) without convergence.")

        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters:
        - X: Data array of shape (n_samples, n_features).

        Returns:
        - labels: Array of cluster labels.
        """
        X = self._check_array(X)
        U = self._calculate_membership_matrix_new_data(X)
        labels = np.argmax(U, axis=1)
        return labels

    def _check_array(self, X):
        # for our own ease...
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.values
        else:
            X = np.array(X)
        return X

    def _initialize_U(self):
        """
        Initialize the membership matrix U with random values that sum to 1 for each sample.
        """
        np.random.seed(self.random_state)
        U = np.random.dirichlet(np.ones(self.n_clusters), size=self.n_samples)
        self.U = U

    def _calculate_cluster_centers(self):
        """
        Calculate the cluster centers based on the current membership matrix.
        """
        um = self.U ** self.m
        centers = um.T @ self.X / np.sum(um.T, axis=1, keepdims=True)
        return centers

    def _calculate_distances(self, X, centers):
        """
        Calculate the squared Euclidean distance from each data point to each cluster center.

        Parameters:
        - X: Data array.
        - centers: Cluster centers.

        Returns:
        - distances: Distance matrix.
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, center in enumerate(centers):
            distances[:, i] = np.linalg.norm(X - center, axis=1) ** 2
        return distances

    def _update_membership_matrix(self):
        """
        Update the membership matrix U with suppression.
        """
        distances = self._calculate_distances(self.X, self.centers)
        # Avoid division by zero
        distances = np.fmax(distances, 1e-10)

        # Calculate the initial membership matrix without suppression
        exponent = 2 / (self.m - 1)
        temp = distances ** (-exponent)
        denominator = np.sum(temp, axis=1, keepdims=True)
        U_new = temp / denominator

        # Apply suppression
        U_suppressed = np.zeros_like(U_new)
        for k in range(self.n_samples):
            u_k = U_new[k]
            u_wk = np.max(u_k)
            w = np.argmax(u_k)

            # Compute suppressed winner membership
            mu_wk = np.sin((np.pi * u_wk) / 2) ** self.xi

            # Compute suppressed non-winner memberships
            alpha_k = (1 - mu_wk) / (1 - u_wk + 1e-10)  # Small epsilon to avoid division by zero

            for i in range(self.n_clusters):
                if i == w:
                    U_suppressed[k, i] = mu_wk
                else:
                    U_suppressed[k, i] = alpha_k * u_k[i]

            # Normalize to ensure sum to 1
            U_suppressed[k] /= np.sum(U_suppressed[k])

        return U_suppressed

    def _calculate_membership_matrix_new_data(self, X):
        """
        Calculate the membership matrix U for new data X.

        Parameters:
        - X: New data array.

        Returns:
        - U: Membership matrix.
        """
        distances = self._calculate_distances(X, self.centers)
        # Avoid division by zero
        distances = np.fmax(distances, 1e-10)

        exponent = 2 / (self.m - 1)
        temp = distances ** (-exponent)
        denominator = np.sum(temp, axis=1, keepdims=True)
        U_new = temp / denominator

        # Apply suppression
        U_suppressed = np.zeros_like(U_new)
        for k in range(X.shape[0]):
            u_k = U_new[k]
            u_wk = np.max(u_k)
            w = np.argmax(u_k)

            # Compute suppressed winner membership
            mu_wk = np.sin((np.pi * u_wk) / 2) ** self.xi

            # Compute suppressed non-winner memberships
            alpha_k = (1 - mu_wk) / (1 - u_wk + 1e-10)

            for i in range(self.n_clusters):
                if i == w:
                    U_suppressed[k, i] = mu_wk
                else:
                    U_suppressed[k, i] = alpha_k * u_k[i]

            # Normalize
            U_suppressed[k] /= np.sum(U_suppressed[k])

        return U_suppressed
