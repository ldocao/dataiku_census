import numpy as np
import pandas as pd
import load_data as ld
import ipdb
import matplotlib.pyplot as plt
from constants import *

training_file = "../data/census_income_learn.csv"
metadata_file = "../data/census_income_metadata.txt"
df = ld.prepare_dataframe(training_file, metadata_file=metadata_file)
nrows = len(df)



for c in df.columns.values:
    print c
    plt.figure()   
    try:
        df[c].hist(by=df[PREDICTION_COLNAME], orientation="horizontal")
    except TypeError:
        counts = df[c].groupby(df[PREDICTION_COLNAME]).value_counts()
        false_counts = counts[False]
        true_counts = counts[True]
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        false_counts.plot(kind="barh", ax=ax1)
        true_counts.plot(kind="barh", ax=ax2)
        #ax1.set_xticklabels(ax1.get_xticklabels(),rotation="vertical")
        #ax2.set_xticklabels(ax2.get_xticklabels(),rotation="vertical")


    plt.suptitle(c)
    plt.tight_layout()
    plt.savefig("./figures/histograms_groupby/"+c+".pdf")