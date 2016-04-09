import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from GrbSVM import GrbSVM

def preProcess(train,test):
    training = pd.read_csv(train)
    test = pd.read_csv(test)
    
    training = training.replace(-999999,2)
    test = test.replace(-999999,2)
    
    print(training.shape)
    print(test.shape)
    
    # remove constant columns
    remove = []
    for col in training.columns:
        if training[col].std() == 0:
            remove.append(col)
    
    training.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    
    # remove duplicated columns
    remove = []
    c = training.columns
    
    for i in range(len(c)-1):
        v = training[c[i]].values
        for j in range(i+1,len(c)):
            if np.array_equal(v,training[c[j]].values):
                remove.append(c[j])
    
    training.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)
    
    
    print(training.shape)
    print(test.shape)
    return training,test

process = GrbSVM()
train,test = preProcess("../input/train.csv","../input/test.csv")
process.SVC(train,test);
