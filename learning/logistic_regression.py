import numpy as np
import pandas as pd
import load_data as ld
import feature_engineering as feat
from sklearn.linear_model import LogisticRegression
from constants import *



def predict(df):
    """Return prediction of incomes classification"""

    ## load data
    df = ld.prepare_dataframe(TRAINING_FILE, metadata_file=METADATA_FILE)
    df = feat.engineer_dataframe(df)

    ## machine learning: logistic regression
    cls = LogisticRegression() #define classifier
    features = df.drop(PREDICTION_COLNAME, axis=1)
    target = df[PREDICTION_COLNAME]
    cls.fit(features, target)
    predictions = cls.predict(features)
    
    return predictions




