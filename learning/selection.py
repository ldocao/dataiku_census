import random
import numpy as np
import pandas as pd
import utils


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

    true_selector = df[PREDICION_COLNAME].values == True
    false_selector = df[PREDICION_COLNAME].values == False

    df_true = df[true_selector]
    df_false = df[false_selector]

    n_rows_false = int(round(ratio * len(df_true)))

    ##check the number of selected rows for false
    if n_rows_false > len(df_false): 
        raise Warning("Ratio is not respected. Return max size of False instead.")
        n_rows_selected = len(df_false)
    else:
        n_rows_selected = n_rows_false

    df_false = utils.sample_random(df_false, n_rows_false)

    return concat([df_true, df_false])