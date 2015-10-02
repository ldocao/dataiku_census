import numpy as np
import pandas as pd
import load_data as ld
import ipdb
import matplotlib.pyplot as plt
from constants import PREDICTION_COLNAME
import utils

training_file = "../data/census_income_learn.csv"
metadata_file = "../data/census_income_metadata.txt"




def unique_elements(list_a, list_b):
    """Return the list of unique elements in set_a, and set_b"""
    set_a = set(list_a)
    set_b = set(list_b)
    unique_elements = set_a ^ set_b
    unique_a = unique_elements.intersection(set_a)
    unique_b = unique_elements.intersection(set_b)

    return unique_a, unique_b










df = ld.prepare_dataframe(training_file, metadata_file=metadata_file)
nrows = len(df)
colnames = df.columns.values


is_categorical = np.array([not df[c].is_numeric() for c in colnames])
categorical_variables = list(colnames[is_categorical])
n_nans = df.isnull().sum() #print number of nan per columns
print n_nans/float(nrows) #frequency on NaN


df_true = df[df[PREDICTION_COLNAME].values]
df_false = df[df[PREDICTION_COLNAME].values == False]




compared_colnames = categorical_variables.append("age")
for c in categorical_variables:
    unique_true, unique_false = unique_elements(df_true[c].values, df_false[c].values)
    print "======", c.upper()
    print unique_false
    print unique_true

