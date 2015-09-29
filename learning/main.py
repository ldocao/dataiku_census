import numpy as np
import pandas as pd
import load_data as ld
import visualize
import feature_engineering as feat
from constants import *
from sklearn.metrics import confusion_matrix
import ipdb




def machine_learning(df_train, df_valid, method=METHOD):
    """Return dataframe with prediction 

    Parameters:
    ----------
    df_train: pd.DataFrame
        training df containing features and target 

    df_valid: pd.DataFrame
        validation df containing ONLY features.
    """

    import logistic_regression

    switcher ={
            "logistic_regression": logistic_regression.predict
            #SVM
            #random forest
            #neural network
    }
    
    func = switcher.get(METHOD)
    return func(df_train, df_valid)



def print_score(y_true, y_valid):
    confusion_score = confusion_matrix(y_true, y_valid)
    print confusion_score
    print "Score summary: ", round(float(np.trace(confusion_score))/n_valid, 3)*100., "%"
    # read confusion matrix as follows:
    # (expected true, predicted true) (expected true, predicted false)
    # (expected false, predicted true) (expected false, predicted false)


## LOAD DATA
print "loading data..."
df_train = ld.prepare_dataframe(TRAINING_FILE, metadata_file=METADATA_FILE)
df_train = feat.engineer_dataframe(df_train)

###should make sure here that df_valid has the same features than the training set
df_valid = ld.prepare_dataframe(VALIDATION_FILE, metadata_file=METADATA_FILE)
df_valid = df_valid[["age", "education", "sex", "num persons worked for employer", PREDICTION_COLNAME]]
df_valid = feat.dummify(df_valid, "education") 
n_valid = len(df_valid) 


## LEARNING
print "training: "+METHOD+" ..."
prediction = machine_learning(df_train, df_valid.drop(PREDICTION_COLNAME, axis=1), method=METHOD)

## MEASURE OF SUCCESS, PLOT CONTROL
print_score(df_valid[PREDICTION_COLNAME].values, prediction)

#visualize.compare_results(prediction, validation)