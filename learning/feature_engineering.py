import random
import numpy as np
import pandas as pd
import utils
from constants import *
import ipdb
import visualize


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
    return df.drop(columns_drop, axis=1)
    


def dummify(df, colname):
    """Return a dataframe in which colname has been dummified

    Parameters:
    ----------
    df: pd.DataFrame

    colname: string
        name of column in df to be dummify. cannot be a list
    """

    dummies = pd.get_dummies(df[colname])
    #dummies.drop(dummies.columns[0], axis=1, inplace=True) #remove arbitrarely the first column to avoid colinearity problem
    df.drop(colname, inplace=True, axis=1)
    return pd.concat([df, dummies], axis=1)



def dummify_all_categorical(df):
    """Return a dataframe where every categorical variables have been dummified"""

    df = pd.get_dummies(df)
    df = dummify(df, "detailed industry recode")
    df = dummify(df, "detailed occupation recode") ## add some variables that are encoded as int64 but that are in fact categorical
    return df


def engineer(train, valid):
    """Apply some feature engineering on dataframes"""
    train = dummify_all_categorical(train)
    valid = dummify_all_categorical(valid)
    train, valid = utils.common_columns(train, valid) #keep only common features

    return train, valid







