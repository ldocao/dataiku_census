import random
import numpy as np
import pandas as pd
import utils
from constants import *
import ipdb

def bias_population(df, ratio, seed=0):
    """Return a biased sample of the dataframe.

    Parameters:
    ----------
    df: pd.dataframe

    ratio: float
        ratio of True vs False for "incomes classification".

    seed: integer
        random seed for selecting false rows

    Output:
    ------
    result: pd.dataframe

    Comments:
    --------
    The number of lines for true is always maximised for a given ratio.
    """

    true_selector = df[PREDICTION_COLNAME].values == True
    false_selector = df[PREDICTION_COLNAME].values == False

    df_true = df[true_selector]
    df_false = df[false_selector]

    n_rows_false = int(round(ratio * len(df_true)))

    ##check the number of selected rows for false
    if n_rows_false > len(df_false): 
        raise Warning("Ratio is not respected. Return max size of False instead.")
        n_rows_selected = len(df_false)
    else:
        n_rows_selected = n_rows_false

    df_false = utils.sample_random(df_false, n_rows_false, seed=seed)

    return pd.concat([df_true, df_false])



def drop_high_nan(df, threshold=0.5):
    """Drop columns if number of nan is greater than threshold"""
    n_nans =  df.isnull().sum()
    freq_nans = n_nans/float(len(df)) #in percentage
    to_drop = (freq_nans > threshold).values
    columns_drop = df.columns.values[to_drop].tolist()
    df.drop(columns_drop, inplace=True, axis=1)
    return df