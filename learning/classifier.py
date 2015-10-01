import numpy as np
import pandas as pd



def logistic_regression(features_train, target_train, features_valid):
    """Return prediction based on features_train of validation set using logistic regression

    Parameters:
    ----------
    features_train: pd.DataFrame
        training set containing only the features

    target_train: pd.DataFrame
        training set containing only the target

    features_valid: pd.DataFrame
        validation set containing only the features

    Output:
    ------
    result: np.array
    """

    from sklearn.linear_model import LogisticRegression

    cls = LogisticRegression()
    cls.fit(features_train, target_train)
    return cls.predict(features_valid)





def random_forest(features_train, target_train, features_valid):
    """Return prediction based on features_train of validation set using random forest

    Parameters:
    ----------
    features_train: pd.DataFrame
        training set containing only the features

    target_train: pd.DataFrame
        training set containing only the target

    features_valid: pd.DataFrame
        validation set containing only the features

    Output:
    ------
    result: np.array
    """
    from sklearn.ensemble import RandomForestClassifier as RandomForest

    cls = RandomForest()
    cls.fit(features_train, target_train)
    return cls.predict(features_valid)




def predict(features_train, target_train, features_valid):
    """Call the suitable function to predict on validation set"""

    from constants import CLASSIFIER
    
    switcher ={
        "logistic_regression": logistic_regression,
        "random_forest": random_forest
        #SVM
        #random forest
        #neural network
    }
    print "CLASSIFIER: ", CLASSIFIER
    func = switcher.get(CLASSIFIER)
    return func(features_train, target_train, features_valid)