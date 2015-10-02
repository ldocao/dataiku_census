import numpy as np
import pandas as pd
import random
from constants import *
import ipdb

def replace_character(serie, c, r):
    """Return the same dataframe where "c" is replaced by "r" in colname

    Parameters:
    ----------
    serie: pd.Series

    c: string
        character to replace

    r: string
        character to replace by

    Output:
    ------
    df: pd.dataframe
    """

    return serie.map(lambda x: x.lstrip(c).rstrip(r))


def sample_random(df, n, seed=0):
    """Return N random rows of df

    Parameters:
    ----------
    df: pd.dataframe

    n: integer
    """
    random.seed(0)
    return df.ix[random.sample(df.index, n)]



def is_numeric(self):
    """Is pd.Series a numerical variable"""
    return self.dtypes in NUMERICS

setattr(pd.Series, 'is_numeric', is_numeric) # add method to pd.Series


def common_columns(df1,df2):
    """Return the common columns between df1 and df2

    Parameters:
    ----------
    df1, df2: pd.DataFrame
    """

    set1 = set(df1.columns.values)
    set2 = set(df2.columns.values)
    unique_col = list(set1 ^ set2)

    for c in unique_col:
        if c in df1.columns:
            df1.drop(c, inplace=True, axis=1)
        else:
            df2.drop(c, inplace=True, axis=1)

    return df1, df2
        










