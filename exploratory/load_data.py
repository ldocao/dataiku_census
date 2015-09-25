import numpy as np
import pandas as pd
import utils
from constants import *
import ipdb




def generate_colname(metadata_file):
    """Return a list of names to be used to label training dataframe 

    Parameters:
    ----------
    metadata_file: string
        full path to metadata.txt file

    Output:
    colnames: list
        list of names (and associated code) for training dataframe column
    """

    metadata = pd.read_csv(metadata_file, comment="|", sep=":", names=["names","values"])
    names = metadata["names"].tolist()
    names.pop(0) #remove first dummy element
    names.append(PREDICTION_COLNAME) #add what we want to predict
    return names




def _remove_leading_space(df):
    """Return the data frame after removing leading spaces for all elements"""
    return df.replace(to_replace="^\s",value="",regex=True)



def _clean(df):
    """Return a cleaned version of data frame

    """

    df.replace(to_replace="Not in universe", value=UNKNOWN, inplace=True)
    df[PREDICTION_COLNAME].replace(to_replace="- 50000.", value=False, inplace=True)
    df[PREDICTION_COLNAME].replace(to_replace="50000+.", value=True, inplace=True)
    ##add more clean up here
    return df

def prepare_dataframe(training_file, metadata_file=False, **kwargs):
    """Return a data frame with cleaned column name

    Parameters:
    ----------
    training_file: string
        full path to training set as csv

    metadata_file: string
        full path to metadata.txt file to generate column names in resulting dataframe

    kwargs: 
        optional keywords for read_csv
    """


    ## generate column names
    if metadata_file:
        colnames = generate_colname(metadata_file)
    else:
        colnames = None

    ## load data frame
    df = pd.read_csv(training_file, names=colnames)

    ## clean up
    df = _remove_leading_space(df)
    df = _clean(df)

    return df




