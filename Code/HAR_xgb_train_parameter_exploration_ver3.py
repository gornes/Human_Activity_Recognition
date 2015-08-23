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


y_train = y_train - 1
y_test = y_test - 1
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test, label=y_test)

xgb_parameters = {}
xgb_parameters['objective'] = 'multi:softmax'
xgb_parameters['silent'] = 0
xgb_parameters['nthread'] = 4
xgb_parameters['num_class'] = 6 
xgb_parameters['eval_metric'] = 'merror'

num_round = 10

# 1. XGB General Accuracy as function of eta (feature weights shrinkage) parameter
eta = np.arange(0, 1, 0.1)
accuracies = []
for n in eta:
    tot = 0
    xgb_parameters['eta'] = n
    for i in xrange(5):
	xgbg = xgb.train(xgb_parameters, xg_train, num_round) 
        y_pred = xgbg.predict(xg_test)
	tot += accuracy_score(y_test, y_pred)
    accuracies.append(tot / 5)
    print n, accuracies
plt.plot(eta, accuracies)
plt.xlabel('eta')
plt.ylabel('Accuracy')
plt.title('XGB General accuracy as function of eta') 
plt.show()


# 2. XGB General Accuracy as function of max_depth parameter
max_depth = range(5, 50, 5)
accuracies = []
for n in max_depth:
    tot = 0
    xgb_parameters['max_depth'] = n
    for i in xrange(5):
	xgbg = xgb.train(xgb_parameters, xg_train, num_round) 
        y_pred = xgbg.predict(xg_test)
	tot += accuracy_score(y_test, y_pred)
    accuracies.append(tot / 5)
    print n, accuracies
plt.plot(max_depth, accuracies)
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('XGB General accuracy as function of max_depth') 
plt.show() 



  

   
