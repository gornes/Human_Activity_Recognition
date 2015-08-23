import matplotlib.pyplot as plt
%matplotlib inline

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler


rfc = RandomForestClassifier()
rfc.get_params()

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

# 1. RFC Accuracy as function of number of trees (n_estimators)
#num_trees = range(5, 50, 5)
num_trees = [5, 10, 15, 20, 25, 30, 50, 100, 1000]
accuracies = []
for n in num_trees:
    tot = 0
    for i in xrange(5):
        rf = RandomForestClassifier(n_estimators=n)
        rf.fit(X_train, y_train)
        tot += rf.score(X_test, y_test)
    accuracies.append(tot / 5)
    print n, accuracies
plt.plot(num_trees, accuracies)
plt.xlabel('n_stimators')
plt.ylabel('Accuracy')
plt.title('RFC accuracy as function of n_estimators')
plt.show() 

# 2. RFC Accuracy as function of max_features parameter
#num_features = range(1, n_features + 1)
num_features = [ 1, 2, 3, 4, 5, 9, 24, 100, 250, n_features] 
accuracies = []
for n in num_features:
    tot = 0
    for i in xrange(5):
        rf = RandomForestClassifier(max_features=n)
        rf.fit(X_train, y_train)
        tot += rf.score(X_test, y_test)
    accuracies.append(tot / 5)
    print n, accuracies
plt.plot(num_features, accuracies)
plt.xlabel('max_features')
plt.ylabel('Accuracy')
plt.title('RFC accuracy as function of max_features') 
plt.show()

# 3. RFC Accuracy as function of max_depth parameter
max_depth = range(2, 21, 2)
accuracies = []
for n in max_depth:
    tot = 0
    for i in xrange(5):
        rf = RandomForestClassifier(max_depth=n)
        rf.fit(X_train, y_train)
        tot += rf.score(X_test, y_test)
    accuracies.append(tot / 5)
    print n, accuracies
plt.plot(max_depth, accuracies)
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('RFC accuracy as function of max_depth') 
plt.show()

# 4. RFC Accuracy as function of min_samples_split parameter
min_samples_split = range(2, 21, 2)
accuracies = []
for n in min_samples_split:
    tot = 0
    for i in xrange(5):
        rf = RandomForestClassifier(min_samples_split=n)
        rf.fit(X_train, y_train)
        tot += rf.score(X_test, y_test)
    accuracies.append(tot / 5)
    print n, accuracies
plt.plot(min_samples_split, accuracies)
plt.xlabel('min_samples_split_')
plt.ylabel('Accuracy')
plt.title('RFC accuracy as function of min_samples_split_') 
plt.show()

# 5. RFC Accuracy as function of min_samples_split parameter
min_samples_leaf = range(2, 21, 2)
accuracies = []
for n in min_samples_leaf:
    tot = 0
    for i in xrange(5):
        rf = RandomForestClassifier(min_samples_leaf=n)
        rf.fit(X_train, y_train)
        tot += rf.score(X_test, y_test)
    accuracies.append(tot / 5)
    print n, accuracies
plt.plot(min_samples_leaf, accuracies)
plt.xlabel('min_samples_split_')
plt.ylabel('Accuracy')
plt.title('RFC accuracy as function of min_samples_split_') 
plt.show()

# 6. RFC Feature Importance graph
rfc = RandomForestClassifier(n_estimators=20, criterion='gini', max_depth=8, min_samples_split=6, min_samples_leaf=8, 
            min_weight_fraction_leaf=0.0, max_features=24, max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=-1, 
            random_state=None, verbose=0, warm_start=False, class_weight=None)
rfc.fit(X_train, y_train)
fi_mean = np.mean(rfc.feature_importances_)
print 'Feature importance mean:', fi_mean
fi_sorted_index = np.argsort(rfc.feature_importances_)
print "Top 10 features are:", list(X_train_df.columns[fi_sorted_index[-1:-11:-1]])
index = np.arange(n_features)
plt.bar(index, rfc.feature_importances_[fi_sorted_index[::-1]], color='b',)
plt.xlabel('features')
plt.ylabel('Feature Importance')
plt.title('RFC Feature Importance Plot') 
plt.show()

index = np.arange(50)
plt.bar(index, rfc.feature_importances_[fi_sorted_index[-1:-51:-1]], color='b',)
plt.xlabel('features')
plt.ylabel('Feature Importance')
plt.title('RFC Feature Importance Plot') 
plt.show()



