import numpy as np
import pandas as pd
import random
from constants import *

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