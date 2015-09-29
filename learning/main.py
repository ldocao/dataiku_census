import numpy as np
import pandas as pd
import load_data as ld
import visualize
import feature_engineering as feat
from constants import *
import ipdb

def machine_learning(df_train, df_valid, method="logistic_regression"):
    """Return dataframe with prediction


    """

    import logistic_regression

    switcher ={
            "logistic_regression": logistic_regression.predict
            #SVM
            #random forest
            #neural network
    }
    
    func = switcher.get(method)
    return func(df_train, df_valid)



## PARAMETERS
method = "logistic_regression"

## LOAD DATA
df_train = ld.prepare_dataframe(TRAINING_FILE, metadata_file=METADATA_FILE)
df_train = feat.engineer_dataframe(df_train)
df_valid = ld.prepare_dataframe(VALIDATION_FILE, metadata_file=METADATA_FILE)
df_valid = feat.engineer_dataframe(df_valid)


## LEARNING
prediction = machine_learning(df_train, df_valid.drop(PREDICTION_COLNAME), method=method)

## PLOT CONTROL
#visualize.compare_results(prediction, validation)