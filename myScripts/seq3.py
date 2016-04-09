
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
t0 = time()
np.random.seed(42)

training = pd.read_csv("../input2/final.csv")

trainNoLabel = training.drop(['label'], axis=1).values

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    
bench_k_means(KMeans(init='k-means++', n_clusters=14, n_init=10),
              name="k-means++", data=trainNoLabel)

print time()-t0