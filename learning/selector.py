import numpy as np
import pandas as pd
from constants import *






def variance_threshold(features_train, features_valid):
    """Return the initial dataframes after dropping some features according to variance threshold

    Parameters:
    ----------
    features_train: pd.DataFrame
        features of training set

    features_valid: pd.DataFrame
        features of validation set

    Output:
    ------
    features_train: pd.DataFrame

    features_valid: pd.DataFrame
    """
    from sklearn.feature_selection import VarianceThreshold    

    threshold=0.01
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(features_train)

    ## Instead of using the transform() method, we look at which columns have been dropped, to be able to drop in both training and validation set the same features. This way, we keep the column names to make interpretation easier
    variances = selector.variances_
    dropped_features = features_train.columns.values[variances < threshold] #name of features to drop
    features_train.drop(dropped_features, axis=1, inplace=True)
    features_valid.drop(dropped_features, axis=1, inplace=True)

    return features_train, features_valid




def reduce_dimension(features_train, features_valid):
    """Call the suitable function to reduce dimensionality of datasets"""

    switcher ={
        "variance_threshold": variance_threshold,
        #PCA
    }
    
    func = switcher.get(SELECTOR)
    return func(features_train, features_valid)