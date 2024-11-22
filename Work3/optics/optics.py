from sklearn.cluster import OPTICS

def opticsAlgorithm(X, metric, algorithm, min_samples, n_jobs=-1): #cluster_method, min_samples, eps,
    optics = OPTICS(metric=metric,
                    algorithm=algorithm,
                    min_samples=min_samples,#cluster_method=cluster_method, eps=eps,
                    n_jobs=n_jobs)
    return optics.fit_predict(X)