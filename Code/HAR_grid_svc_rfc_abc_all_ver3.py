import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV


def get_data(X_train_file, y_train_file, X_test_file, y_test_file):
    # Reads train & test features and labels data from files and returnes them as numpy arrays

    X_train_df = pd.read_csv(X_train_file)
    y_train_df = pd.read_csv(y_train_file)
    X_test_df = pd.read_csv(X_test_file)
    y_test_df = pd.read_csv(y_test_file);

    n_train_samples = X_train_df.shape[0]
    n_test_samples = X_test_df.shape[0]
    n_features = X_train_df.shape[1]

    X_train = np.array(X_train_df).reshape((n_train_samples,n_features))
    y_train = np.array(y_train_df).reshape(n_train_samples,)    
    X_test = np.array(X_test_df).reshape((n_test_samples,n_features))
    y_test = np.array(y_test_df).reshape(n_test_samples,)

    return X_train, y_train, X_test, y_test


def get_scores(y_test,y_pred):
    # Reads labels and predictions and gives accuracy, precision, recall & confusion matrix

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    prec = np.around(np.diag(cm).astype(float)*100/cm.sum(axis = 0), decimals =2)
    rec = np.around(np.diag(cm).astype(float)*100/cm.sum(axis = 1), decimals =2)

    cm_full = np.vstack((cm,prec))  # adding precision row 
    cm_full = np.hstack((cm_full,(np.append(rec,np.around(acc*100,decimals=2))).reshape(len(cm_full),1))) # adding recall column & total accuracy

    prec_macro = precision_score(y_test, y_pred, average='weighted')
    rec_macro = recall_score(y_test, y_pred, average='weighted')

    print 'Accuracy: ', np.around(acc*100,decimals=2)
    print 'Precision: ', round(np.mean(prec),2)
    print 'Recall: ', round(np.mean(rec),2)
    print 'Macro Precision: ', round(prec_macro*100,2)
    print 'Macro Recall: ', round(rec_macro*100,2)   
    print 'Confusion Matrix (Activities: Walking, Upstairs, Downstairs, Standing, Sitting, Laying'
    print cm
    print 'Confusion Matrix & Scores (Actual Activities & Precision vs. Predicted Activies & Recall; Total Accuracy)'
    print cm_full  

    return acc, prec_macro, rec_macro, cm, cm_full


def do_grid_search(est, parameters, X_train, y_train):
    # Reads estimator and it's parameters and gives the best parameters

    nfolds = 10
    skf = cross_validation.StratifiedKFold(y_train, n_folds = nfolds, random_state=42)
    gs_clf = GridSearchCV(est, parameters, cv = skf, n_jobs = -1)
    gs_clf.fit(X_train, y_train)
    return gs_clf.best_score_, gs_clf.best_params_
 

def do_svc(X_train, y_train, X_test, svc_parameters):
    # Read data and give SVC prediction for the best parameters

    svc = svm.SVC()
    svc_best_score, svc_best_params = do_grid_search(svc, svc_parameters, X_train, y_train)
    print 'SVC best score is: ', svc_best_score
    print 'SVC best parameters are: ', svc_best_params

    svc_opt = svm.SVC(C=svc_best_params['C'], kernel=svc_best_params['kernel'], degree=3, gamma=svc_best_params['gamma'], 
            coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
            max_iter=-1, random_state=None)
    svc_opt.fit(X_train, y_train)
    return svc_opt.predict(X_test)


def do_rfc(X_train, y_train, X_test, rfc_parameters):
    # Read data and give RFC prediction for the best parameters

    rfc = RandomForestClassifier()
    rfc_best_score, rfc_best_params = do_grid_search(rfc, rfc_parameters, X_train, y_train)
    print 'RFC best score is: ', rfc_best_score
    print 'RFC best parameters are: ', rfc_best_params


    rfc_opt = RandomForestClassifier(n_estimators=rfc_best_params['n_estimators'], criterion=rfc_best_params['criterion'], 
            max_depth=rfc_best_params['max_depth'], min_samples_split=2, min_samples_leaf=rfc_best_params['min_samples_leaf'], min_weight_fraction_leaf=0.0,            
            max_features=rfc_best_params['max_features'], max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=-1, 
            random_state=None, verbose=0, warm_start=False, class_weight=None)
    rfc_opt.fit(X_train, y_train)

    feature_importances = np.argsort(rfc_opt.feature_importances_) 
    print 'RFC: 10 most importantant features are with column numbers: ', feature_importances[-1:-11:-1] # reverse order

    return rfc_opt.predict(X_test)


def do_abc(X_train, y_train, X_test, abc_parameters):
    # Read data and give Ada Boost Classification prediction for the best parameters

    abc = AdaBoostClassifier()
    abc_best_score, abc_best_params = do_grid_search(abc, abc_parameters, X_train, y_train)
    print 'ABC best score is: ', abc_best_score
    print 'ABC best parameters are: ', abc_best_params

    abc_opt = AdaBoostClassifier((DecisionTreeClassifier(max_depth=2)), n_estimators=abc_best_params['n_estimators'], 
              learning_rate=abc_best_params['learning_rate'])
    abc_opt.fit(X_train, y_train)

    feature_importances = np.argsort(abc_opt.feature_importances_) 
    print 'ABC: 10 most importantant features are with column numbers: ', feature_importances[-1:-11:-1] # reverse order

    return abc_opt.predict(X_test)


if __name__ == '__main__':

    # Get Data
    X_train, y_train, X_test, y_test = get_data('X_train_header.txt', 'y_train_header.txt', 'X_test_header.txt', 'y_test_header.txt')
'''    
    # Get GridSearch best SVC Model & it's metrics
    C_range = [0.1, 0.316, 1, 3.16, 10, 31.6, 100, 316, 1000, 3160, 10000]
    gamma_range = [1.0000e-08, 6.3096e-08, 3.9811e-07, 2.5119e-06, 1.5849e-05, 0.0001, 0.00063096,
                            0.0039811, 0.025119, 0.15849, 1.0000]                           
    svc_parameters = {'kernel':('linear', 'rbf'), 'C':C_range, 'gamma': gamma_range}
    y_pred_svc = do_svc(X_train, y_train, X_test, svc_parameters)
    print 'GridSearch SVC Metrics:'
    svc_acc, svc_prec, svc_rec, svc_cm, svc_cm_full = get_scores(y_test,y_pred_svc)
'''
    # Get GridSearch best RFC Model  & it's metrics
    n_estimators_range = [100, 500, 1000]
    max_features_range = [9, 24, 561]    # log2(nfeatures=562) = 24, sqrt(561) = 9     
    max_depth_range = [None, 6, 8] 
    min_samples_leaf_range = [1, 4, 6]
    rfc_parameters = {'criterion':('gini', 'entropy'), 'n_estimators': n_estimators_range, 'max_features': max_features_range,
                      'max_depth': max_depth_range, 'min_samples_leaf': min_samples_leaf_range}
    y_pred_rfc = do_rfc(X_train, y_train, X_test, rfc_parameters)
    print 'GridSearch RFC Metrics: '
    rfc_acc, rfc_prec, rfc_rec, rfc_cm, rfc_cm_full = get_scores(y_test,y_pred_rfc)

    # Get GridSearch best ABC Model  & it's metrics
    n_estimators_range = [5, 10, 25, 50, 100, 500, 1000]
    learning_rate_range = [0.1, 0.3, 0.5, 0.7, 0.9, 1]   
    abc_parameters = {'n_estimators': n_estimators_range, 'learning_rate': learning_rate_range}
    y_pred_abc = do_abc(X_train, y_train, X_test, abc_parameters)
    print 'GridSearch ABC Metrics: '
    abc_acc, abc_prec, abc_rec, abc_cm, abc_cm_full = get_scores(y_test,y_pred_abc)

    # Boosting SVC, RFC & ABC
    y_all = np.clumn_stack((y_pred_svc, y_pred_rfc, y_pred_abc))
    y_pred_all = (stats.mode(y_all)[0][0]).astype(int)
    print 'All Metrics: '
    all_acc, all_prec, all_rec, all_cm, all_cm_full = get_scores(y_test,y_pred_all)

