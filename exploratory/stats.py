import numpy as np
import pandas as pd
import load_data as ld
import ipdb


training_file = "../data/census_income_learn.csv"
metadata_file = "../data/census_income_metadata.txt"

result = ld.prepare_dataframe(training_file, metadata_file=metadata_file)


for c in result.columns.values:
    print result[c]
    ipdb.set_trace()