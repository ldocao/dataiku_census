##function which project onto a 2D plane with name of two axis as arguments to plot the difference between prediction and validation
import pandas as pd
import numpy as np
from constants import *
import utils
import matplotlib.pyplot as plt
import ipdb
import seaborn as sns




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
    return num


def _categorical_to_numerical(x):
    """Return a numerical conversion as pd.Series from categorical

    Parameters:
    ----------
    x: pd.Series

    Output:
    ------
    result: pd.Series
    """

    num = _map_categorical(x)
    return x.map(num)



def scatter(df, colx, coly, colhue):
    """Scatter plot for either numerical or categorical variables

    Parameters:
    ----------
    df: pd.DataFrame

    colx: string
        name of x column in df to use for abcisse

    coly : string
        name of y column in df to use for ordinates

    colhue: string
        name of column to use for color
    """

    plt.figure()
    sns.stripplot(x=colx, y=coly, hue=colhue, data=df);
    plt.show()











