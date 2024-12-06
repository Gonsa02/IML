from sklearn.cluster import SpectralClustering

def spectralAlgorithm(X, eigen_solver, affinity, n_neighbors, assign_labels, n_clusters, seed, n_jobs=-1):
    spectral = SpectralClustering(eigen_solver=eigen_solver,
                                  affinity=affinity,
                                  n_neighbors=n_neighbors,
                                  assign_labels=assign_labels,
                                  n_clusters=n_clusters,
                                  random_state=seed,
                                  n_jobs=n_jobs)
    
    return spectral.fit_predict(X)