from sklearn import svm

def svmAlgorithm(X_train, Y_train, X_test, **kwargs):
    # Filter out parameters that are not accepted by svm.SVC
    valid_params = svm.SVC().get_params()
    clf_params = {k: v for k, v in kwargs.items() if k in valid_params}

    clf = svm.SVC(**clf_params)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    return predictions
