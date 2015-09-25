import numpy as np
import pandas as pd
import load_data as ld
import ipdb
import matplotlib.pyplot as plt

training_file = "../data/census_income_learn.csv"
metadata_file = "../data/census_income_metadata.txt"
df = ld.prepare_dataframe(training_file, metadata_file=metadata_file)
nrows = len(df)

for c in df.columns.values:
    print c
    plt.figure()
    try:
        df[c].hist(orientation="horizontal")
    except TypeError:
        df[c].value_counts().plot(kind='barh') 

    plt.axvline(nrows,linestyle="dashed",color="black")
    plt.xlim([0,nrows])
    plt.suptitle(c)
    plt.tight_layout()
    plt.savefig("./figures/histograms/"+c+".pdf")