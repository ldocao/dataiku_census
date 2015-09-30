import numpy as np
import pandas as pd
import load_data as ld
import visualize
import utils
import feature_engineering as feat
from constants import *
import ipdb



def print_score(y_true, y_valid):
    """Display some score measurement"""
    from sklearn.metrics import confusion_matrix

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
target_train = train[PREDICTION_COLNAME]
target_valid = valid[PREDICTION_COLNAME]



from sklearn.feature_selection import VarianceThreshold
variance_threshold=0.01
selector = VarianceThreshold(threshold=variance_threshold)
selector.fit(features_train)
variances = selector.variances_
dropped_features = features_train.columns.values[variances < variance_threshold] #name of features to drop
features_train.drop(dropped_features, axis=1, inplace=True)
features_valid.drop(dropped_features, axis=1, inplace=True)






from sklearn.linear_model import LogisticRegression
cls = LogisticRegression()
cls.fit(features_train, target_train)
prediction = cls.predict(features_valid)





## MEASURE OF SUCCESS, PLOT CONTROL
print_score(target_valid, prediction)

#visualize.compare_results(prediction, validation)















