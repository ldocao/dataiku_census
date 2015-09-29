import numpy as np
import pandas as pd
import load_data as ld
import visualize



def machine_learning(df, method="logistic_regression"):
    """Return dataframe with prediction"""

    import logistic_regression as lr

    switcher ={
            "logistic_regression": lr.predict(df)
    }
    
    func = switcher.get(method)
    return func(df)



## PARAMETERS
method = "logistic_regression"
training_file = "../data/census_income_learn.csv"
metadata_file = "../data/census_income_metadata.txt"
validation_file = "../data/census_income_test.csv"



## LOAD DATA
df = ld.prepare_dataframe(training_file, metadata_file=metadata_file)
validation = ld.prepare_dataframe(validation_file, metadata_file=metadata_file)

## LEARNING
prediction = machine_learning(df, method=method)

## PLOT CONTROL
visualize.compare_results(prediction, validation)