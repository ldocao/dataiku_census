##function which project onto a 2D plane with name of two axis as arguments to plot the difference between prediction and validation
import pandas as pd
import numpy as np
from constants import *
import utils
import matplotlib.pyplot as plt
import ipdb





def _map_categorical(x):
    """Return a dictionary between categorical and numerical variable

    Parameters:
    ----------
    x: pd.Series

    Output:
    ------
    result: dict
        map between categorical to numerical variable
    """
    alpha = np.unique(x) #x is any type
    num = pd.Series(range(len(alpha)), index=alpha) #numerical equivalent to set
    return num.to_dict()


def _categorical_to_numerical(x):
    """Return a numerical conversion as pd.Series from categorical

    Parameters:
    ----------
    x: pd.Series

    Output:
    ------
    result: pd.Series
    """
    alpha = np.unique(x) #x is any type
    num = pd.Series(range(len(alpha)), index=alpha) #numerical equivalent to set
    ipdb.set_trace()
    return x.map(num)



def scatter(x, y):
    """Scatter plot for either numerical or categorical variables

    Parameters:
    ----------
    x, y : pd.Series
    """

    if not x.is_numeric(): x = _categorical_to_numerical(x)
    if not y.is_numeric(): y = _categorical_to_numerical(y)

    df = pd.DataFrame()
    df["x"] = x
    df["y"] = y

    plt.figure()
    pd.set_option('display.mpl_style', 'default')
    ax = plt.axes()
    df.plot(x='x', y='y', kind='scatter', ax=ax, alpha=0.6)
    plt.show()











    