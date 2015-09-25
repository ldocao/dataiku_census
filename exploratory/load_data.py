import numpy as np
import pandas as pd
import utils
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
    names.append("incomes classification") #add what we want to predict
    return names





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


    if metadata_file:
        colnames = generate_colname(metadata_file)
    else:
        colnames = None

    return pd.read_csv(training_file, names=colnames)



