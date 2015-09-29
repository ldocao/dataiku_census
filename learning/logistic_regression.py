import numpy as np
import pandas as pd
import load_data as ld
import feature_engineering as feat
from sklearn.linear_model import LogisticRegression
from constants import *



def predict(df_training, df_validation):
    """Return prediction of incomes classification

    Parameters:
    ----------
    df_training: pd.DataFrame
        training df containing features and target 

    df_validation: pd.DataFrame
        validation df containing ONLY features.

    Output:
    ------
    predictions: np.array (len(df_validation))
    """

    cls = LogisticRegression() #define classifier
    features = df_training.drop(PREDICTION_COLNAME, axis=1)
    target = df_training[PREDICTION_COLNAME]
    cls.fit(features, target)
    predictions = cls.predict(df_validation)
    
    return predictions




