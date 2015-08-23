import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn import cross_validation

from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points
from moe.optimal_learning.python.data_containers import SamplePoint


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


def xgboost_train_cross_validation(X_train, y_train, params, num_round, cv_folds):
    # Get data and gives XGB prediction using cross validation
    labels_cv = []
    pred_cv = []
    for train_index, test_index in cv_folds:
        X_train_cv = X_train[train_index, :]
        y_train_cv = y_train[train_index]
        X_test_cv = X_train[test_index, :]
        y_test_cv = y_train[test_index]
        
        xg_train_cv = xgb.DMatrix(X_train_cv, label=y_train_cv)
        xg_test_cv = xgb.DMatrix(X_test_cv, label=y_test_cv)
        
        bst = xgb.train(params, xg_train_cv, num_round)
        pred = bst.predict(xg_test_cv)
        pred_cv.extend(pred)
        labels_cv.extend(y_test_cv)        
    return get_scores(labels_cv, pred_cv)


def xgboost_cross_validation(X_train, y_train, params, cv_folds):
    # Get data and gives XGB prediction using cross validation
    labels_cv = []
    pred_cv = []
    for train_index, test_index in cv_folds:
        X_train_cv = X_train[train_index, :]
        y_train_cv = y_train[train_index]
        X_test_cv = X_train[test_index, :]
        y_test_cv = y_train[test_index]
        
    	xgbc = xgb.XGBClassifier(max_depth=params['max_depth'], learning_rate=params['learning_rate'], 
			     n_estimators=params['n_estimators'], silent=True, objective="multi:softmax",
			     nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
			     base_score=0.5, seed=0, missing=None)       
        xgbc.fit(X_train_cv, y_train_cv)
        pred = xgbc.predict(X_test_cv)
        pred_cv.extend(pred)
        labels_cv.extend(y_test_cv)        
    return get_scores(labels_cv, pred_cv)        


def do_xgboost_train(X_train, y_train, X_test, y_test):
    # Get data and gives XGB prediction using cross validation for searching for best parameters
    y_train = y_train - 1
    y_test = y_test - 1

    n_folds = 10
    cv_folds = cross_validation.StratifiedKFold(y_train, n_folds=n_folds)

    xgb_parameters = {}
    xgb_parameters['objective'] = 'multi:softmax'
    xgb_parameters['silent'] = 0
    xgb_parameters['nthread'] = 4
    xgb_parameters['num_class'] = 6 
    xgb_parameters['eval_metric'] = 'merror'
    num_round = 10

    # Finding the best XGBoost eta &max_depth parameters
    best_acc = 0.0
    best_params =[]
    for eta in np.arange(0.1, 1.2, .5):
       for max_depth in np.arange(5, 106, 5):
            xgb_parameters['eta'] = eta
            xgb_parameters['max_depth'] = max_depth
            acc, prec, rec, cm, cm_full = xgboost_train_cross_validation(X_train, y_train, xgb_parameters, num_round, cv_folds)
            print eta, max_depth, acc
            if acc > best_acc:
                best_acc = acc
                best_params = [eta, max_depth]
    
    xgb_parameters['eta'] = best_params[0]
    xgb_parameters['max_depth'] = best_params[1]
    print 'XGBoost train best eta & max_depth: ', best_params 
    print 'XGBoost train best accuracy: ', best_acc  

    # Finding best xgboost prediction 
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)

    xgb_bst = xgb.train(xgb_parameters, xg_train, num_round)
    # get prediction
    return xgb_bst.predict(xg_test) + 1


def do_xgboost(X_train, y_train, X_test, y_test):
    # Get data and gives XGB prediction using cross validation for searching for best parameters
    y_train = y_train - 1
    y_test = y_test - 1

    n_folds = 10
    cv_folds = cross_validation.StratifiedKFold(y_train, n_folds=n_folds)

    xgb_parameters = {}
    # Finding the best XGBoost eta &max_depth parameters
    best_acc = 0.0
    best_params =[]
    for learning_rate in np.arange(0.1, 1.1, 0.1):
        for n_estimators in [100, 500, 1000]:
            for max_depth in [3, 4, 5]:
                xgb_parameters['learning_rate'] = learning_rate
                xgb_parameters['n_estimators'] = n_estimators
                xgb_parameters['max_depth'] = max_depth
                acc, prec, rec, cm, cm_full = xgboost_cross_validation(X_train, y_train, xgb_parameters, cv_folds)
                print learning_rate, n_estimators, max_depth, acc
                if acc > best_acc:
                    best_acc = acc
                    best_params = [learning_rate, n_estimators, max_depth]
    
    xgb_parameters['learning_rate'] = best_params[0]
    xgb_parameters['n_estimators'] = best_params[1]
    xgb_parameters['max_depth'] = best_params[2]
    print 'XGBoost best learning_rate, n_estimators & max_depth: ', best_params 
    print 'XGBoost best best accuracy: ', best_acc  

    # Finding best xgboost prediction 
    xgbc = xgb.XGBClassifier(max_depth=xgb_parameters['max_depth'], learning_rate=xgb_parameters['learning_rate'], 
	                       n_estimators=xgb_parameters['n_estimators'], silent=True, objective="multi:softmax",
	                       nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
	                       base_score=0.5, seed=0, missing=None)
    xgbc.fit(X_train, y_train)
    # get prediction
    return xgbc.predict(X_test) + 1

def do_xgb_train_MOE(num_points_to_sample, X_train, y_train, verbose=True, **kwargs):
    # Finding Best XGB parameters using MOE
    xgb_parameters = {}
    xgb_parameters['objective'] = 'multi:softmax'
    xgb_parameters['silent'] = 1
    xgb_parameters['nthread'] = 4
    xgb_parameters['num_class'] = 6  
    # Range of XGBoost parameters that are optimized
    exp_xgb = Experiment([[0.1, 1], [0.02, 1]])  # eta_range = [0.1, 1]; max_depth_range = [2, 100] but it is normalized

    num_round = 5
    n_folds = 10
    cv_folds = cross_validation.StratifiedKFold(y_train, n_folds=n_folds)

    best_point = []
    best_point_value = 0.
    for _ in range(num_points_to_sample):
        # Use MOE to determine what is the point with highest Expected Improvement to use next
        next_point_to_sample = gp_next_points(exp_xgb, rest_host='localhost', rest_port=6543, **kwargs)[0]  # By default we only ask for one point

        # Sample the point from objective function
        xgb_parameters['eta'] = next_point_to_sample[0]
        xgb_parameters['max_depth'] = int(round(next_point_to_sample[1]*100))          
        acc_cv, prec_cv, rec_cv, cm_cv, cm_full_cv = xgboost_train_cross_validation(X_train, y_train, xgb_parameters, num_round, cv_folds)
        value_of_next_point = acc_cv
        if value_of_next_point > best_point_value:
            best_point_value = value_of_next_point
            best_point = next_point_to_sample
        if verbose:
            print "Sampled f({0:s}) = {1:.18E}".format(str(next_point_to_sample), value_of_next_point)
        # Add the information about the point to the experiment historical data to inform the GP
        exp_xgb.historical_data.append_sample_points([SamplePoint(next_point_to_sample, -value_of_next_point, 0.0001)])  # We can add some noise
    best_point[1] = int(round(best_point[1] * 100))
    return best_point, best_point_value


def do_xgb_MOE(num_points_to_sample, X_train, y_train, verbose=True, **kwargs):
    # Finding Best XGB parameters using MOE
    xgb_parameters = {}
    # Range of XGBoost parameters that are optimized
    exp_xgb = Experiment([[0.1, 1], [0.002, 1]], [0.01, 1])  # learning_rate_range = [0.1, 1]; n_estimators_range = [2, 1000] is normalized
                                                            # max_depth_range = [1, 100] is normalized

    n_folds = 10
    cv_folds = cross_validation.StratifiedKFold(y_train, n_folds=n_folds)

    best_point = []
    best_point_value = 0.
    for _ in range(num_points_to_sample):
        # Use MOE to determine what is the point with highest Expected Improvement to use next
        next_point_to_sample = gp_next_points(exp_xgb, rest_host='localhost', rest_port=6543, **kwargs)[0]  # By default we only ask for one point

        # Sample the point from objective function
        xgb_parameters['learning_rate'] = next_point_to_sample[0]
        xgb_parameters['n_estimators'] = int(round(next_point_to_sample[1]*1000))   
        xgb_parameters['max_depth'] = int(round(next_point_to_sample[2]*100))       
        acc_cv, prec_cv, rec_cv, cm_cv, cm_full_cv = xgboost_cross_validation(X_train, y_train, xgb_parameters, cv_folds)
        value_of_next_point = acc_cv
        if value_of_next_point > best_point_value:
            best_point_value = value_of_next_point
            best_point = next_point_to_sample
        if verbose:
            print "Sampled f({0:s}) = {1:.18E}".format(str(next_point_to_sample), value_of_next_point)
        # Add the information about the point to the experiment historical data to inform the GP
        exp_xgb.historical_data.append_sample_points([SamplePoint(next_point_to_sample, -value_of_next_point, 0.0001)])  # We can add some noise
    best_point[1] = int(round(best_point[1] * 1000))
    best_point[2] = int(round(best_point[2] * 100))
    return best_point, best_point_value


if __name__ == '__main__':

    # Get Data
    X_train, y_train, X_test, y_test = get_data('X_train_header.txt', 'y_train_header.txt', 'X_test_header.txt', 'y_test_header.txt')
    
    # Get XGBoost train Model metrics
    y_pred_xgb = do_xgboost_train(X_train, y_train, X_test, y_test)
    print 'XGBoost Train Metrics: '
    xgb_acc, xgb_prec, xgb_rec, xgb_cm, xgb_cm_full = get_scores(y_test,y_pred_xgb)


    # Get XGBoost Model metrics
    y_pred_xgbc = do_xgboost(X_train, y_train, X_test, y_test)
    print 'XGBoost Classification Metrics: '
    xgbc_acc, xgbc_prec, xgbc_rec, xgbc_cm, xgbc_cm_full = get_scores(y_test,y_pred_xgbc)


    # Get MOE best XGB train Model  & it's metrics
    num_points_to_sample = 100   
    num_round = 10    

    y_train = y_train - 1
    y_test = y_test - 1
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)


    xgb_parameters = {}
    xgb_parameters['objective'] = 'multi:softmax'
    xgb_parameters['silent'] = 1
    xgb_parameters['nthread'] = 4
    xgb_parameters['num_class'] = 6  


    params_xgb_moe = do_xgb_train_MOE(num_points_to_sample, X_train, y_train)
    xgb_parameters['eta'] = params_xgb_moe[0]
    xgb_parameters['max_depth'] = params_xgb_moe[1]

    xgb_moe = xgb.train(xgb_parameters, xg_train, num_round)
    y_pred_xgb_moe = xgb_moe.predict(xg_test) + 1
    print 'XGBoost Train MOE Metrics: '
    xgb_moe_acc, xgb_moe_prec, xgb_moe_rec, xgb_moe_cm, xgb_moe_cm_full = get_scores(y_test + 1, y_pred_xgb_moe)


    # Get MOE best XGB Model  & it's metrics
    num_points_to_sample = 100   

    y_train = y_train - 1
    y_test = y_test - 1

    xgb_parameters = {}

    params_xgb_moe = do_xgb_MOE(num_points_to_sample, X_train, y_train)
    xgb_parameters['learning_rate'] = params_xgb_moe[0]
    xgb_parameters['n_estimators'] = params_xgb_moe[1]
    xgb_parameters['max_depth'] = params_xgb_moe[2]

    xgbc_moe = xgb.XGBClassifier(max_depth=xgb_parameters['max_depth'], learning_rate=xgb_parameters['learning_rate'], 
			     n_estimators=xgb_parameters['n_estimators'], silent=True, objective="multi:softmax",
			     nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
			     base_score=0.5, seed=0, missing=None)       
    xgbc_moe.fit(X_train, y_train)
    y_pred_xgbc_moe = xgbc_moe.predict(X_test) + 1
    
    print 'XGBoost Classification MOE Metrics: '
    xgbc_moe_acc, xgbc_moe_prec, xgbc_moe_rec, xgbc_moe_cm, xgbc_moe_cm_full = get_scores(y_test + 1, y_pred_xgbc_moe)
