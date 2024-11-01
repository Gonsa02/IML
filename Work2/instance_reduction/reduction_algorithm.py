from .drop3 import DROP3
from .ennth import ENNTh
from .gcnn import GCNN

def reductionAlgorithm(X_train, Y_train, method):
    """
    Applies the specified instance reduction method to the training data.

    Parameters:
    - X_train (pd.DataFrame): Training features.
    - Y_train (pd.Series): Training labels.
    - method (str): The instance reduction method to apply ('drop3', 'ennth', 'gcnn').

    Returns:
    - X_reduced (pd.DataFrame): Reduced training features.
    - Y_reduced (pd.Series): Reduced training labels.
    """
    if method == "drop3":
        ir_model = DROP3(X_train, Y_train)
    elif method == "ennth":
        ir_model = ENNTh(X_train, Y_train)
    elif method == "gcnn":
        ir_model = GCNN(X_train, Y_train)
    else:
        raise ValueError(f"Unknown instance reduction method: {method}")
    
    return ir_model.fit()