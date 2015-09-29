##function which project onto a 2D plane with name of two axis as arguments to plot the difference between prediction and validation
import pandas as pd
import numpy as np

def _categorical_to_numerical(x):
    """Return a numerical conversion as pd.Series from categorical

    Parameters:
    ----------
    x: pd.Series

    Output:
    ------
    result: pd.Series
    """

    alpha = set(x) #any type
    num = pd.Series(range(len(alpha)), index=alpha) #numerical equivalent to set
    return x.map(num)



def scatter(x, y):
    """Scatter plot for either numerical or categorical variables

    Parameters:
    ----------
    x, y : pd.Series
    """

    df["x"] = x
    df["y"] = y

    plt.figure()
    pd.set_option('display.mpl_style', 'default')
    ax = plt.axes()
    df.plot(x='x', y='y', kind='scatter', ax=ax, alpha=0.6)
    plt.show()