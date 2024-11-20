from sklearn.cluster import OPTICS

def opticsAlgorithm(X, metric, algorithm, n_jobs=-1): #cluster_method, min_samples, eps,
    optics = OPTICS(metric=metric,
                    algorithm=algorithm,
                    #cluster_method=cluster_method, min_samples=min_samples, eps=eps,
                    n_jobs=n_jobs)
    return optics.fit_predict(X)