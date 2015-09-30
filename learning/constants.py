import numpy as np

UNKNOWN = np.nan
PREDICTION_COLNAME = "incomes classification"
NUMERICS = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


TRAINING_FILE = "../data/census_income_learn.csv"
METADATA_FILE = "../data/census_income_metadata.txt"
VALIDATION_FILE = "../data/census_income_test.csv" 

FEATURE_SELECTION = "variance_threshold"
CLASSIFIER = "logistic_regression"
