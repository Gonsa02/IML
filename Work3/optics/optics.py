from sklearn.cluster import OPTICS

def opticsAlgorithm(X, metric, algorithm, min_samples, n_jobs=-1):
    optics = OPTICS(metric=metric,
                    algorithm=algorithm,
                    min_samples=min_samples,
                    n_jobs=n_jobs)
    
    return optics.fit_predict(X)