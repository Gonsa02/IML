from sklearn.cluster import OPTICS
from sklearn import neighbors

def opticsAlgorithm(X, metric, algorithm, cluster_method, min_samples, n_jobs=-1):
    optics = OPTICS(metric=metric,
                    algorithm=algorithm,
                    cluster_method=cluster_method,
                    min_samples=min_samples,
                    n_jobs=n_jobs)
    return optics.fit_predict(X)