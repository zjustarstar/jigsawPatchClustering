import numpy as np
from sklearn.feature_selection import VarianceThreshold

def vt_selection(features:np.ndarray):
    """
    Perform feature selection using VarianceThreshold.

    Args:
        features (numpy.ndarray): The input feature matrix as a NumPy array, where each row
                                  represents a data point and each column represents a feature.

    Returns:
        numpy.ndarray: A modified feature matrix after applying VarianceThreshold feature selection.

    Example:
        To perform feature selection on a feature matrix 'input_features', you can call the function as follows:
        selected_features = vt_selection(input_features)
    """
    
    selector = VarianceThreshold() # Default 0
    vt_features=selector.fit_transform(features)
    return vt_features