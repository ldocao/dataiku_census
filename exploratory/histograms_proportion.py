#PURPOSE: this is just another way of looking at histograms_groupby

import numpy as np
import pandas as pd
import load_data as ld
import ipdb
import matplotlib.pyplot as plt
from constants import *
import utils

training_file = "../data/census_income_learn.csv"
metadata_file = "../data/census_income_metadata.txt"
df = ld.prepare_dataframe(training_file, metadata_file=metadata_file)
nrows = len(df)


## list numerical vs categorical variables
colnames =  df.columns.values
is_numerical = np.array([df[c].is_numeric() for c in colnames])
is_categorical = np.logical_not(is_numerical)
numerical_variables = list(colnames[is_numerical])
numerical_variables.remove("detailed industry recode")
numerical_variables.remove("detailed occupation recode")
categorical_variables = list(colnames[is_categorical])
categorical_variables.append("detailed industry recode")
categorical_variables.append("detailed occupation recode")



for c in df.columns.values:

    plt.figure()   
    if c in categorical_variables:
        counts = df[c].groupby(df[PREDICTION_COLNAME]).value_counts()
        false_counts = counts[False]
        true_counts = counts[True]
        false_counts /= float(false_counts.sum()) #normalize to total number of persons
        true_counts /= float(true_counts.sum()) #normalize to total number of persons
        normalized_counts = pd.DataFrame({"50000+":true_counts, "-50000":false_counts})

        print "categorical: ", c
        normalized_counts.plot(kind="barh", stacked=True)
    elif c in numerical_variables:
        counts = df[c].groupby(df[PREDICTION_COLNAME])
        false_counts = counts.get_group(False)
        true_counts = counts.get_group(True)
        plt.hist([false_counts,true_counts],
                  stacked=True, color=["r","g"],
                  bins=100,normed=True)
        print "numerical: ",c
    else:
        print c
        raise KeyError("Not an existing categorical nor a numerical variable")

    plt.suptitle(c)
    plt.tight_layout()
    plt.savefig("./figures/histograms_proportion/"+c+".pdf")
    ipdb.set_trace()