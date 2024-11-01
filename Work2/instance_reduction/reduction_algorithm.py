from .drop3 import DROP3
from .ennth import ENNTh
from .gcnn import GCNN

def reductionAlgorithm(X_train, Y_train, method, **kwargs):
    """
    Applies the specified instance reduction method to the training data.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - Y_train (pd.Series): Training labels.
    - method (str): The instance reduction method to apply ('drop3', 'ennth', 'gcnn').
    - **kwargs: Additional keyword arguments for the instance reduction method.

    Returns:
    - X_reduced (pd.DataFrame): Reduced training features.
    - Y_reduced (pd.Series): Reduced training labels.
    """
    # Check if any element in X_train or Y_train is a string
    if X_train.applymap(type).eq(str).any().any():
        print("Non-numeric data detected in X_train:")
        print("X_train:")
        print(X_train)
        raise ValueError("X_train should only contain numeric data.")

    if Y_train.apply(lambda x: isinstance(x, str)).any():
        print("Non-numeric data detected in Y_train:")
        print("Y_train:")
        print(Y_train)
        raise ValueError("Y_train should only contain numeric data.")

    if method == "drop3":
        metric = kwargs.get('metric', 'minkowski_r2')
        voting = kwargs.get('voting', 'majority')
        k = kwargs.get('k', 3)
        ir_model = DROP3(X_train, Y_train, metric=metric, voting=voting, k=k)
    elif method == "ennth":
        ir_model = ENNTh(X_train, Y_train)
    elif method == "gcnn":
        ir_model = GCNN(X_train, Y_train)
    else:
        raise ValueError(f"Unknown instance reduction method: {method}")
    
    return ir_model.fit()