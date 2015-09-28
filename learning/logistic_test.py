import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


n_points = 1000
df = pd.DataFrame({'x': np.random.normal(0, 1, n_points),
                   'y': np.random.normal(0, 1, n_points)})

df["v"] = 2*df.x - df.y
probabilities = 1 / (1 + np.exp(-1 * df.v))

x = - 1/3 * (df['x'] + df['y'])
probabilities = 1 / (1 + np.exp(-1 * x))
df['target'] = np.random.uniform(0, 1, n_points) < probabilities


plt.figure()
df.plot(kind='scatter', x='x', y='y', c="target");
plt.show()