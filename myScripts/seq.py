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
t0 = time()
np.random.seed(42)
digits = pd.read_csv("../input2/droidKin.csv")

# data = scale(digits.data)


target =  digits["label"]
n_digits = len(np.unique(target))
labels = target
digits = digits.drop(['label'], axis=1).values
n_samples, n_features = digits.shape

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

print(79 * '_')
print('% 9s' % 'init'
      '    time  inertia    homo   compl  v-meas     ARI AMI  silhouette')

def bench_k_means(estimator, name, data):
    estimator.fit(data)
    
bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=digits)

print time()-t0