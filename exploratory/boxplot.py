import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import load_data as ld


training_file = "../data/census_income_learn.csv"
metadata_file = "../data/census_income_metadata.txt"
df = ld.prepare_dataframe(training_file, metadata_file=metadata_file)


df_numeric = df._get_numeric_data() #remove non-numeric
df_numeric = df_numeric.iloc[:,(df_numeric.dtypes != bool).values] #remove boolean

pd.options.display.mpl_style = 'default'
for c in df_numeric.columns.values:
    plt.figure()
    df_numeric.boxplot(column=c)
    plt.savefig("./figures/boxplot/"+c+".pdf")