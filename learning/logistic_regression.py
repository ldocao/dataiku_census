import numpy as np
import pandas as pd
import load_data as ld
import feature_engineering as feat
from sklearn.linear_model import LogisticRegression
from constants import *



def predict(df):
    """Return prediction of incomes classification"""

    cls = LogisticRegression() #define classifier
    features = df.drop(PREDICTION_COLNAME, axis=1)
    target = df[PREDICTION_COLNAME]
    cls.fit(features, target)
    predictions = cls.predict(features)
    
    return predictions




