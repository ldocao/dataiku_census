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
        name of column in df to be dummify
    """

    dummies = pd.get_dummies(df[colname])
    #dummies.drop(dummies.columns[0], axis=1, inplace=True) #remove arbitrarely the first column to avoid colinearity problem
    df.drop(colname, inplace=True, axis=1)
    return pd.concat([df, dummies], axis=1)



def dummify_all_categorical(df):
    """Return a dataframe where every categorical variables have been dummified"""

    df = pd.get_dummies(df)
    df = dummify(df, ["detailed industry recode", "detailed occupation recode"]) ## add some variables that are encoded as int64 byt that are in fact categorical

    return df



def select_by_pca(df, retain_ratio=0.99, plot_variance=True):
    """Return the dataframe"""

    from sklearn.decomposition import PCA

    ## make a first pass to get the variance as function of components
    pca_obj = PCA()
    min_comp = 10
    df_trans = pca_obj.fit_transform(df) 
    s = pca_obj.explained_variance_ratio_
    sum=0.0
    comp=0

    for _ in s:
        sum += _
        comp += 1
        if(sum>=retain_ratio):
            break


    if comp < min_comp: #take at least min_comp components
        comp = min_comp

    print 'Number of selected components to retain: ', comp 

    if plot_variance:
        visualize.variance_explained(s, comp)


    ## now take only the desired number of components
    pca_obj = PCA(n_components=comp)
    #newdf = pca_obj.fit_transform(df)
    X = pca_obj.fit_transform(df)   
    newdf = pca_obj.inverse_transform(X) 
    ##problem here I want to get back my df, should we use MDA ? Multiple Discriminant Analysis. http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
    ipdb.set_trace()
    return newdf




def engineer_dataframe(df):
    """Return a dataframe engineered for machine learning"""

    df = bias_population(df, 5.)
    df = dummify_all_categorical(df)
    df = select_by_pca(df.drop(PREDICTION_COLNAME, axis=1))


    #df = drop_high_nan(df)

    return df





