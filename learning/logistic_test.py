
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import ipdb
import visualize


def scatter_two_categories(cat1, cat2):
    pd.set_option('display.mpl_style', 'default')
    ax = plt.axes()
    cat1.plot(x='x', y='y', kind='scatter', ax=ax, alpha=0.6, label='True')
    cat2.plot(x='x', y='y', kind='scatter', color='orange', ax=ax, alpha=0.6, label='False')
    return ax


##PREPARE DATA
### features
n_points = 100000
np.random.seed(0)
y_list = ["a", "b", "c"]
y_dict = {"a":1, "b":2, "c":3}
df = pd.DataFrame({'x': np.random.normal(0, 1, n_points),
                   'y': np.random.choice(y_list,n_points)})

### target classification
yy = np.array([y_dict[i] for i in df["y"].values])
x = - 2./3 * (df['x'].values + yy) 
probabilities = 1 / (1 + np.exp(-1 * x))
df['target'] = np.random.uniform(0, 1, n_points) < probabilities
target = df['target']





# ipdb.set_trace()


# ###subselect sample to plot
# trues = df[['x', 'y']][target][:500]
# falses = df[['x', 'y']][~target][:500]
# plt.figure()
# pd.set_option('display.mpl_style', 'default')
# ax = plt.axes()
# trues.plot(x='x', y='y', kind='scatter', ax=ax, alpha=0.6, label='True')
# falses.plot(x='x', y='y', kind='scatter', color='orange', ax=ax, alpha=0.6, label='False')
# plt.suptitle("Training set")
# plt.show()



# ## LOGISTIC REGRESSION
# cls = LogisticRegression() #define classifier
# features = df[['x', 'y']]
# features_train, features_test, target_train, target_test = train_test_split(features, df['target'])
# cls.fit(features_train, target_train)
# predictions = cls.predict(features_test)

# df_test = features_test
# df_test["predictions"] = predictions
# trues = df_test[['x', 'y']][predictions][:500]
# falses = df_test[['x', 'y']][~predictions][:500]
# plt.figure()
# ax = plt.axes()
# pd.set_option('display.mpl_style', 'default')
# trues.plot(x='x', y='y', kind='scatter', ax=ax, alpha=0.6, label='True')
# falses.plot(x='x', y='y', kind='scatter', ax=ax, color='orange', alpha=0.6, label='False')
# plt.suptitle("Validation set")
# plt.show()


# print "finished"