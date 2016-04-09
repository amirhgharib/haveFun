print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import cluster, datasets
t0 = time()
np.random.seed(42)

data = pd.read_csv("../input2/mobomarket.csv")
k_means = cluster.KMeans(init='k-means++')
k_means.fit(data) 

print max(k_means.labels_[:29000])

print time()-t0