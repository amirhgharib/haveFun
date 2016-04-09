import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.preprocessing import LabelEncoder

class GrbSVM:
    def __init__(self):
        end=1
    
    def SVC(self,train,test):
        print "Start SVM..."
        y_train = train['TARGET'].values
        X_train = train.drop(['ID','TARGET'], axis=1).values
        id_test = test['ID']
        X_test = test.drop(['ID'], axis=1).values
        
        model = SVC().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print y_pred
        #scores = model.decision_function(test.head(100))
        #predictions = 1.0 / (1.0 + np.exp(-scores))
        #row_sums = predictions.sum(axis=1)
        
        # create submission file
        submission = pd.DataFrame({"ID": id_test, "TARGET": y_pred})
        submission.to_csv("SVC_Submission.csv", index=False)
        
        print('SVC Completed!')
        