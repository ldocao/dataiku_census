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




def turn_binary(serie, true_value, false_value):
    """Return a pandas series converting two values into boolean

    Parameters:
    ----------
    serie: pd.series

    true_value: object
        value to be converted as True

    false_value: object
        value to be converted as False
    """

    serie.replace(to_replace=true_value, value=True, inplace=True)
    serie.replace(to_replace=false_value, value=False, inplace=True)
    return serie



def _clean(df):
    """Return a cleaned version of data frame

    """

    df.replace(to_replace="Not in universe", value=UNKNOWN, inplace=True)
    df.replace(to_replace="?", value=UNKNOWN, inplace=True)
    df[PREDICTION_COLNAME] = turn_binary(df[PREDICTION_COLNAME], "- 50000.", "50000+.")
    df["fill inc questionnaire for veteran's admin"] = turn_binary(df["fill inc questionnaire for veteran's admin"], "Yes", "No")
    df["member of a labor union"] = turn_binary(df["member of a labor union"], "Yes", "No")
    df["migration prev res in sunbelt"] = turn_binary(df["migration prev res in sunbelt"], "Yes", "No")
    df["sex"] = turn_binary(df["sex"], "Male", "Female")
    df["year"] = turn_binary(df["year"], 94, 95)

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




