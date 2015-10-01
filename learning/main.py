import numpy as np
import pandas as pd
import load_data as ld
from constants import *
import visualize
import utils
import selector
import classifier
import feature_engineering as feat

import ipdb



def print_score(y_true, y_valid):
    """Display some score measurement"""
    from sklearn.metrics import confusion_matrix

    confusion_score = confusion_matrix(y_true, y_valid)
    print confusion_score
    print "Score summary: ", round(float(np.trace(confusion_score))/len(y_valid), 3)*100., "%"
    # read confusion matrix as follows:
    # true = earn 50000+
    # (expected false, predicted false) (expected false, predicted true)
    # (expected true, predicted false) (expected true, predicted true)







## LOAD DATA
print "loading data..."
### basic operation on load data
train = ld.prepare_dataframe(TRAINING_FILE, metadata_file=METADATA_FILE)
valid = ld.prepare_dataframe(VALIDATION_FILE, metadata_file=METADATA_FILE)
train, valid = feat.engineer(train,valid) #pre-process data

### shortcuts
features_train = train.drop(PREDICTION_COLNAME, axis=1)
features_valid = valid.drop(PREDICTION_COLNAME, axis=1)
target_train = train[PREDICTION_COLNAME]
target_valid = valid[PREDICTION_COLNAME]



## SELECT FEATURES
### this is supposed to be step1 in sklearn pipeline, but pipeline bugs with python-2.7
print "selecting features..."
features_train, features_valid = selector.reduce_dimension(features_train, features_valid)


### PREDICT
print "learning parameters and predicting target..."
prediction = classifier.predict(features_train, target_train, features_valid)


## MEASURE OF SUCCESS, PLOT CONTROL
print_score(target_valid, prediction)
#visualize.compare_results(prediction, validation)















