import numpy as np
import pandas as pd

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


def sample_random(df, n):
    """Return N random rows of df

    Parameters:
    ----------
    df: pd.dataframe

    n: integer
    """
    return df.ix[random.sample(df.index, n)]