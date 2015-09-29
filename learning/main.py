import numpy as np
import pandas as pd
import load_data as ld
import visualize



def machine_learning(df, method="logistic_regression"):
    """Return dataframe with prediction"""

    import logistic_regression as lr

    switcher ={
            "logistic_regression": lr.predict(df)
            #SVM
            #random forest
            #neural network
    }
    
    func = switcher.get(method)
    return func(df)



## PARAMETERS
method = "logistic_regression"

## LOAD DATA
df = ld.prepare_dataframe(TRAINING_FILE, metadata_file=METADATA_FILE)
df = feat.engineer_dataframe(df)

## LEARNING
prediction = machine_learning(df, method=method)

## PLOT CONTROL
#visualize.compare_results(prediction, validation)