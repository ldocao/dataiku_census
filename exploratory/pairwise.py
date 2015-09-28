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
