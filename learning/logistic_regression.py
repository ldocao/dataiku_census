import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import load_data as ld
import selection as sel
from sklearn.linear_model import LogisticRegression

training_file = "../data/census_income_learn.csv"
metadata_file = "../data/census_income_metadata.txt"
df = ld.prepare_dataframe(training_file, metadata_file=metadata_file)
df = sel.bias_population(df,1.)



df_subset = df[["age", "instance weight", "incomes classification"]]
incomes = df_subset["incomes classification"]

trues = df_subset[['age', 'instance weight']][incomes][:2000]
falses = df_subset[['age', 'instance weight']][~incomes][:2000]
plt.figure()
pd.set_option('display.mpl_style', 'default')
ax = plt.axes()
trues.plot(x='age', y='instance weight', kind='scatter', ax=ax, alpha=0.3, label='True')
falses.plot(x='age', y='instance weight', kind='scatter', color='orange', ax=ax, alpha=0.3, label='False')
plt.suptitle("Training set")
plt.show()



cls = LogisticRegression() #define classifier
features = df_subset[['age', 'instance weight']]
cls.fit(features, incomes)
predictions = cls.predict(features)

trues = df_subset[['age', 'instance weight']][predictions][:2000]
falses = df_subset[['age', 'instance weight']][~predictions][:2000]
plt.figure()
ax = plt.axes()
pd.set_option('display.mpl_style', 'default')
trues.plot(x='age', y='instance weight', kind='scatter', ax=ax, alpha=0.5, label='True')
falses.plot(x='age', y='instance weight', kind='scatter', color='orange', ax=ax, alpha=0.5, label='False')
plt.suptitle("Prediction")
plt.show()