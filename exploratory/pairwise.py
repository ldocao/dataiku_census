import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import load_data as ld


training_file = "../data/census_income_learn.csv"
metadata_file = "../data/census_income_metadata.txt"
df = ld.prepare_dataframe(training_file, metadata_file=metadata_file)



dpi=150
figsize=(64, 48)
plt.figure(figsize=figsize, dpi=dpi)
axes = pd.tools.plotting.scatter_matrix(df, alpha=0.02, figsize=figsize)
plt.tight_layout()
plt.savefig('./figures/pairwise.png', dpi=dpi)



## list numerical vs categorical variables
colnames =  df.columns.values
is_numerical = np.array([df[c].is_numeric() for c in colnames])
is_categorical = np.logical_not(is_numerical)
numerical_variables = list(colnames[is_numerical])
numerical_variables.remove("detailed industry recode")
numerical_variables.remove("detailed occupation recode")


dpi=150
figsize=(64, 48)
plt.figure(figsize=figsize, dpi=dpi)
axes = pd.tools.plotting.scatter_matrix(df[numerical_variables], alpha=0.02, figsize=figsize)
plt.tight_layout()
plt.savefig('./figures/pairwise_numerical.png', dpi=dpi)