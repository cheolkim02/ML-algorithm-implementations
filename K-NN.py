# import sys
# import subprocess
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib'])
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

x, y = make_blobs(50, n_features=2, centers=2, random_state=0)
knn = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform', metric = 'euclidean')
knn.fit(x, y)