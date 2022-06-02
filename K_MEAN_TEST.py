from random import random, shuffle
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeans

X, y = make_blobs(centers=4, n_samples=500, n_features=2,
                  shuffle=True, random_stae=42)
print(X.shape)


clusters = len(np.unique(y))
print(clusters)


k = KMeans(k=cluster, max_iters=150, plot_steps=False)
y_pred = k.predict(X)


k.plot()

#EX2
from random import random, shuffle
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeans

X, y = make_blobs(centers=4, n_samples=500, n_features=2,
                  shuffle=True, random_stae=42)
print(X.shape)


clusters = len(np.unique(y))
print(clusters)


k = KMeans(k=cluster, max_iters=150, plot_steps=True)
y_pred = k.predict(X)



