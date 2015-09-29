import numpy as np
import pandas as pd
import load_data as ld
import ipdb
import matplotlib.pyplot as plt
import seaborn as sns

training_file = "../data/census_income_learn.csv"
metadata_file = "../data/census_income_metadata.txt"
df = ld.prepare_dataframe(training_file, metadata_file=metadata_file)
nrows = len(df)

##get the number of nan in columns
n_nans = df.isnull().sum() #print number of nan per columns
print n_nans/float(nrows)
