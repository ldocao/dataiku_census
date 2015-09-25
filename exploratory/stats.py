import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



training_file = "../data/census_income_learn.csv"
metadata_file = "../data/census_income_metadata_modified.txt"

df = pd.read_csv(training_file, nrows=1000)
metadata = pd.read_csv(metadata_file, names=["class", "code"], sep='\t')