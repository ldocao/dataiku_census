import numpy as np
import pandas as pd
import load_data as ld
import visualize
import utils
import feature_engineering as feat
from constants import *
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
import ipdb



def print_score(y_true, y_valid):
    """Display some score measurement"""
    confusion_score = confusion_matrix(y_true, y_valid)
    print confusion_score
    print "Score summary: ", round(float(np.trace(confusion_score))/len(y_valid), 3)*100., "%"
    # read confusion matrix as follows:
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
target_valid = valid[PREDICTION_COLNAME].values








from sklearn.linear_model import LogisticRegression
cls = LogisticRegression()
features_train = train.drop(PREDICTION_COLNAME, axis=1)
target_train = train[PREDICTION_COLNAME]
cls.fit(features_train, target_train)
prediction = cls.predict(features_valid)





## MEASURE OF SUCCESS, PLOT CONTROL
print_score(target_valid, prediction)

#visualize.compare_results(prediction, validation)















