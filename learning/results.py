import numpy as np
import pandas as pd
import load_data as ld
import ipdb
import matplotlib.pyplot as plt
from constants import PREDICTION_COLNAME

training_file = "../data/census_income_learn.csv"
metadata_file = "../data/census_income_metadata.txt"
df = ld.prepare_dataframe(training_file, metadata_file=metadata_file)
colnames = df.columns.values
is_categorical = [nps.is_numeric() for s in df]

df_true = df[df[PREDICTION_COLNAME].values]
df_false = df[df[PREDICTION_COLNAME].values == False]



def unique_elements(list_a, list_b):
    """Return the list of unique elements in set_a, and set_b"""
    set_a = set(list_a)
    set_b = set(list_b)
    unique_elements = set_a ^ set_b
    unique_a = unique_elements.intersection(set_a)
    unique_b = unique_elements.intersection(set_b)

    return unique_a, unique_b




for c in colnames:
    unique_true, unique_false = unique_elements(df_true[c].values, df_false[c].values)
    print "======", c
    print unique_true
    print unique_false
