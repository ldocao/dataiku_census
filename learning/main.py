import numpy as np
import pandas as pd
import load_data as ld
import visualize
import feature_engineering as feat
from constants import *
from sklearn.metrics import confusion_matrix
import ipdb



def print_score(y_true, y_valid):
    """Display some score measurement"""
    confusion_score = confusion_matrix(y_true, y_valid)
    print confusion_score
    print "Score summary: ", round(float(np.trace(confusion_score))/len(y_valid), 3)*100., "%"
    # read confusion matrix as follows:
    # (expected false, predicted false) (expected false, predicted true)
    # (expected true, predicted false) (expected true, predicted true)




def predict(train, features_valid):
    """Return prediction of validation from training set

    Parameters:
    ----------
    train: pd.DataFrame
        features and target of training set

    features_valid: pd.DataFrame
        features of validation set

    Output:
    ------
    prediction: np.array [number of rows validation]
    """

    cls = Pipeline([
        ('feature_selection', feat.choose_selecter(), #step 1 in pipeline
        ('classification', feat.choose_classifier()) #step 2 in pipeline
        ])

    features_train = train.drop(PREDICTION_COLNAME, axis=1)
    target_train = train[PREDICTION_COLNAME]
    cls.fit(features_train, target_train) #train the classifier

    return cls.predict(features_valid)






## LOAD DATA
print "loading data..."
### basic operation on load data
train = ld.prepare_dataframe(TRAINING_FILE, metadata_file=METADATA_FILE)
valid = ld.prepare_dataframe(VALIDATION_FILE, metadata_file=METADATA_FILE)

### some feature engineering
train = feat.dummify_all_categorical(train)
valid = feat.dummify_all_categorical(valid)

### shortcuts
features_valid = valid.drop(PREDICTION_COLNAME, axis=1)
target_valid = valid[PREDICTION_COLNAME].values



## PREDICT
print "predicting..."
prediction = predict(train, features_valid)



## MEASURE OF SUCCESS, PLOT CONTROL
print_score(target_valid, prediction)

#visualize.compare_results(prediction, validation)















