from sklearn.cluster import OPTICS

def opticsAlgorithm(X, metric, algorithm):
    optics = OPTICS(metric=metric, algorithm=algorithm)
    return optics.fit_predict(X)