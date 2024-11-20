from sklearn.cluster import SpectralClustering

# Test 'n_neighbors', 'affinity', and 'eigen_solvers'

# label assignment: 'kmeans' and 'cluster_qr

def spectralAlgorithm(X, eigen_solver, affinity, n_neighbors, assign_labels, n_clusters, n_jobs=-1):
    spectral = SpectralClustering(eigen_solver=eigen_solver, affinity=affinity,
                                  n_neighbors=n_neighbors, assign_labels=assign_labels, n_clusters=n_clusters, n_jobs=n_jobs)
    return spectral.fit_predict(X)