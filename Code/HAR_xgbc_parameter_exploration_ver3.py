import matplotlib.pyplot as plt
%matplotlib inline

import pandas as pd
import numpy as np

import xgboost as xgb

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation

from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler   

  
# Reads train & test features and labels data from files and returnes them as numpy arrays
X_train_df = pd.read_csv('X_train_header.txt')
y_train_df = pd.read_csv('y_train_header.txt')
X_test_df = pd.read_csv('X_test_header.txt')
y_test_df = pd.read_csv('y_test_header.txt');

n_train_samples = X_train_df.shape[0]
n_test_samples = X_test_df.shape[0]
n_features = X_train_df.shape[1]

X_train = np.array(X_train_df).reshape((n_train_samples,n_features))
y_train = np.array(y_train_df).reshape(n_train_samples,)    
X_test = np.array(X_test_df).reshape((n_test_samples,n_features))
y_test = np.array(y_test_df).reshape(n_test_samples,)

#xgbc = xgb.XGBClassifier(max_depth=3, learning_rate=params['learning_rate'], 
#			n_estimators=params['n_estimators'], silent=True, objective="multi:softmax",
#			nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
#			base_score=0.5, seed=0, missing=None)  

# 1. XGB Accuracy as function of number of trees parameter
num_trees = range(5, 50, 5)
accuracies = []
for n in num_trees:
    tot = 0
    for i in xrange(5):
	xgbc = xgb.XGBClassifier(n_estimators=n)  
        xgbc.fit(X_train, y_train-1)
	tot += xgbc.score (X_test, y_test-1)
    accuracies.append(tot / 5)
    print n, accuracies
plt.plot(num_trees, accuracies)
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('XGB accuracy as function of n_estimators') 
plt.show()


# 2. XGB Accuracy as function of learning rate parameter
learning_rate = np.arange(0, 1.1, 0.1)
accuracies = []
for n in learning_rate:
    tot = 0
    for i in xrange(5):
        xgbc = xgb.XGBClassifier(learning_rate=n)
        xgbc.fit(X_train, y_train-1)
        tot += xgbc.score(X_test, y_test-1)
    accuracies.append(tot / 5)
    print n, accuracies
plt.plot(learning_rate, accuracies)
plt.xlabel('learning_rate')
plt.ylabel('Accuracy')
plt.title('XGB accuracy as function of learning_rate') 
plt.show() 


# 3. XGB Accuracy as function of max_depth parameter
max_depth = range(2, 11, 2)
accuracies = []
for n in max_depth:
    tot = 0
    for i in xrange(5):
        xgbc = xgb.XGBClassifier(max_depth=n)
        xgbc.fit(X_train, y_train-1)
        tot += xgbc.score(X_test, y_test-1)
    accuracies.append(tot / 5)
    print n, accuracies
plt.plot(max_depth, accuracies)
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('XGB accuracy as function of max_depth') 
plt.show()  
