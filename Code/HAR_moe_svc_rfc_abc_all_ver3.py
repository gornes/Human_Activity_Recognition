import pandas as pd
import numpy as np
from scipy import stats

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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
    rec_macro = recall_score(y_test, y_pred, average='weighted'))

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


def do_svc_linear_MOE(num_points_to_sample, X_train, y_train, verbose=True, **kwargs):
    exp_svc_linear = Experiment([[1.0000e-05, 1.0]])  # C_range = [0.1, 10000] is divided to be in [0.1, 1] range
    best_point = []
    best_point_value = 0.
    for _ in range(num_points_to_sample):
        # Use MOE to determine what is the point with hnighest Expected Improvement to use next
        next_point_to_sample = gp_next_points(exp_svc_linear, rest_host='localhost', rest_port=6543, **kwargs)[0]  # By default we only ask for one point
        # Sample the point from objective function
        C = next_point_to_sample[0] * 10000.0
        svc_linear = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=C, multi_class='ovr',
            fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
        score_cv = cross_validation.cross_val_score(svc_linear, X_train, y_train, cv=10, scoring='accuracy')
        value_of_next_point = np.mean(score_cv) 
        if value_of_next_point > best_point_value:
            best_point_value = value_of_next_point
            best_point = next_point_to_sample
        if verbose:
            print "Sampled f({0:s}) = {1:.18E}".format(str(next_point_to_sample), value_of_next_point)
        # Add the information about the point to the experiment historical data to inform the GP; 
        # - infront of value_of_next_point is due to fact that moe minimize and max accuracy is of interest in HAR classification
        exp_svc_linear.historical_data.append_sample_points([SamplePoint(next_point_to_sample, -value_of_next_point, .000001)])  # We can add some noise
    best_point[0] *= 10000
    return best_point, best_point_value


def do_svc_rbf_MOE(num_points_to_sample, X_train, y_train, verbose=True, **kwargs):
    exp_svc_rbf = Experiment([[1.0000e-05, 1], [1.0000e-08, 1]])  # C_range = [0.1, 10000] is divided to be in [0.1, 1] range
    best_point = []
    best_point_value = 0.
    for _ in range(num_points_to_sample):
        # Use MOE to determine what is the point with highest Expected Improvement to use next
        next_point_to_sample = gp_next_points(exp_svc_rbf, rest_host='localhost', rest_port=6543, **kwargs)[0]  # By default we only ask for one point
        # Sample the point from objective function
        C = next_point_to_sample[0] * 10000.0   
        gamma =   next_point_to_sample[1]  
        svc_rbf = svm.SVC(C=C, kernel='rbf', degree=3, gamma=gamma, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200,
                 class_weight=None, verbose=False, max_iter=-1, random_state=None)
        score_cv = cross_validation.cross_val_score(svc_rbf, X_train, y_train, cv=10, scoring='accuracy')
        value_of_next_point = np.mean(score_cv)
        if value_of_next_point > best_point_value:
            best_point_value = value_of_next_point
            best_point = next_point_to_sample
        if verbose:
            print "Sampled f({0:s}) = {1:.18E}".format(str(next_point_to_sample), value_of_next_point)
        # Add the information about the point to the experiment historical data to inform the GP
        exp_svc_rbf.historical_data.append_sample_points([SamplePoint(next_point_to_sample, -value_of_next_point, 0.0001)])  # We can add some noise
    best_point[0] *= 10000
    return best_point, best_point_value


def do_rfc_MOE(num_points_to_sample, X_train, y_train, verbose=True, **kwargs):
    exp_rfc = Experiment([[0.005, 1], [0.04, 1], [0.1, 1], [0.1, 1]])  # n_estimators_range = [5, 1000] and  max_features_range = [2, 24] are normalized 
                                                                       # max_depth_range = [1, 10] & min_samples_leaf_range = [1, 10] are normalized
    best_point = []
    best_point_value = 0.    
    for _ in range(num_points_to_sample):
        # Use MOE to determine what is the point with highest Expected Improvement to use next
        next_point_to_sample = gp_next_points(exp_rfc, rest_host='localhost', rest_port=6543, **kwargs)[0]  # By default we only ask for one point
        # Sample the point from objective function
        n_estimators = int(round(next_point_to_sample[0] * 1000.0)) 
        max_features =  int(round(next_point_to_sample[1] * 50))  
        max_depth = int(round(next_point_to_sample[2] * 10))    
        min_samples_leaf = int(round(next_point_to_sample[3] * 10))  
        rfc = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', 
            max_depth=max_depth, min_samples_split=2, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=0.0,            
            max_features=max_features, max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=-1, 
            random_state=None, verbose=0, warm_start=False, class_weight=None)
        score_cv = cross_validation.cross_val_score(rfc, X_train, y_train, cv=10, scoring='accuracy')
        value_of next_point =  np.mean(score_cv) 
        if value_of_next_point > best_point_value:
            best_point_value = value_of_next_point
            best_point = next_point_to_sample          
        if verbose:
            print "Sampled f({0:s}) = {1:.18E}".format(str(next_point_to_sample), value_of_next_point)
        # Add the information about the point to the experiment historical data to inform the GP
        exp_rfc.historical_data.append_sample_points([SamplePoint(next_point_to_sample, -value_of_next_point, 0.0001)])  # We can add some noise
    best_point[0] = int(round(best_point[0] * 1000))        
    best_point[1] = int(round(best_point[1] * 50)) 
    best_point[2] = int(round(best_point[2] * 10))  
    best_point[3] = int(round(best_point[3] * 10))    
    return best_point, best_point_value


def do_abc_MOE(num_points_to_sample, X_train, y_train, verbose=True, **kwargs):
    exp_abc = Experiment([[0.005, 1], [0.1, 1]])  # n_estimators_range = [5, 1000] is normalized 
    best_point = []
    best_point_value = 0.    
    for _ in range(num_points_to_sample):
        # Use MOE to determine what is the point with highest Expected Improvement to use next
        next_point_to_sample = gp_next_points(exp_abc, rest_host='localhost', rest_port=6543, **kwargs)[0]  # By default we only ask for one point
        # Sample the point from objective function
        n_estimators = int(round(next_point_to_sample[0] * 1000.0))  
        learning_rate =  next_point_to_sample[1]   
        abc = AdaBoostClassifier((DecisionTreeClassifier(max_depth=2)),n_estimators=n_estimators, learning_rate=learning_rate)
        score_cv = cross_validation.cross_val_score(abc, X_train, y_train, cv=10, scoring='accuracy')
        value_of next_point =  np.mean(score_cv) 
        if value_of_next_point > best_point_value:
            best_point_value = value_of_next_point
            best_point = next_point_to_sample          
        if verbose:
            print "Sampled f({0:s}) = {1:.18E}".format(str(next_point_to_sample), value_of_next_point)
        # Add the information about the point to the experiment historical data to inform the GP
        exp_abc.historical_data.append_sample_points([SamplePoint(next_point_to_sample, -value_of_next_point, 0.0001)])  # We can add some noise
    best_point[0] = int(round(best_point[0] * 1000))        
    return best_point, best_point_value

if __name__ == '__main__':

    # Get Data
    X_train, y_train, X_test, y_test = get_data('X_train_header.txt', 'y_train_header.txt', 'X_test_header.txt', 'y_test_header.txt')
    
    num_points_to_sample = 100
    # Get MOE best SVCLinear Model & it's metrics
    moe_param, moe_accuracy = do_svc_linear_MOE(num_points_to_sample, X_train, y_train)
    print 'Best SVC Linear train param: ', moe_param
    print 'Best SVC Linear train Accuracy: ', moe_accuracy
    C_moe = moe_param[0]
    svc_linear_moe = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=C_moe, multi_class='ovr',
                    fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
    svc_linear_moe.fit(X_train, y_train)
    y_pred_svc_linear_moe = svc_linear_moe.predict(X_test)
    print 'MOE SVCLinear Metrics:'
    print 'Best C: ', C_moe
    svc_linear_moe_acc, svc_linear_moe_prec, svc_linear_moe_rec, svc_linear_moe_cm, svc_linear_moe_cm_full = get_scores(y_test,y_pred_svc_linear_moe)

    # Get MOE best SVC rbf Model & it's metrics
    rbf_moe_param, rbf_moe_accuracy = do_svc_rbf_MOE(num_points_to_sample, X_train, y_train)
    print 'Best SVC RBF train param: ', rbf_moe_param
    print 'Best SVC RBF train Accuracy: ', rbf_moe_accuracy
    C_rbf_moe = rbf_moe_param[0]
    gamma_rbf_moe = rbf_moe_param[1]
    svc_rbf_moe = svm.SVC(C=C_rbf_moe, kernel='rbf', degree=3, gamma=gamma_rbf_moe, 
            coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False,
            max_iter=-1, random_state=None)
    svc_rbf_moe.fit(X_train, y_train)
    y_pred_svc_rbf_moe = svc_rbf_moe.predict(X_test)
    print 'MOE SVC RBF Metrics:'
    print 'Best C_rbf_moe: ', C_rbf_moe 
    print 'Best gamma_rbf_moe: ', gamma_rbf_moe
    svc_rbf_moe_acc, svc_rbf_moe_prec, svc_rbf_moe_rec, svc_rbf_moe_cm, svc_rbf_moe_cm_full = get_scores(y_test,y_pred_svc_rbf_moe)

    # Get MOE best RFC Model  & it's metrics
    rfc_moe_param, rfc_moe_accuracy = do_rfc_MOE(num_points_to_sample, X_train, y_train)
    print 'Best RFC train param: ', rfc_moe_param
    print 'Best RFC train Accuracy: ', rfc_moe_accuracy
    n_estimators = rfc_moe_param[0]
    max_features = rfc_moe_param[1] 
    max_depth = rfc_moe_param[2]
    min_samples_leaf = rfc_moe_param[3]
    rfc_moe = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', 
            max_depth=max_depth, min_samples_split=2, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=0.0,            
            max_features=max_features, max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, 
            random_state=None, verbose=0, warm_start=False, class_weight=None)
    rfc_moe.fit(X_train, y_train)
    y_pred_rfc_moe = rfc_moe.predict(X_test)
    print 'MOE RFC Metrics: '
    print 'Best n_estimators: ', n_estimators
    print 'Best max_features: ', max_features
    print 'Best max_depth: ', max_depth
    print 'Best min_samples_leaf: ' min_samples_leaf
    rfc_moe_acc, rfc_moe_prec, rfc_moe_rec, rfc_moe_cm, rfc_moe_cm_full = get_scores(y_test,y_pred_rfc_moe)


    # Get MOE best ABC Model  & it's metrics
    abc_moe_param, abc_moe_accuracy = do_abc_MOE(num_points_to_sample, X_train, y_train)
    print 'Best ABC train param: ', abc_moe_param
    print 'Best ABC train Accuracy: ', abc_moe_accuracy
    n_estimators = abc_moe_param[0]
    learning_rate = abc_moe_param[1] 
    abc = AdaBoostClassifier((DecisionTreeClassifier(max_depth=2)),n_estimators=n_estimators, learning_rate=learning_rate)
    abc_moe.fit(X_train, y_train)
    y_pred_abc_moe = abc_moe.predict(X_test)
    print 'MOE ABC Metrics: '
    print 'Best n_estimators: ', n_estimators
    print 'Best learning_rate: ', learning_rate
    abc_moe_acc, abc_moe_prec, abc_moe_rec, abc_moe_cm, abc_moe_cm_full = get_scores(y_test,y_pred_abc_moe)

    # Boosting SVC, RFC & ABC
    y_all = np.clumn_stack((y_pred_svc_linear_moe, y_pred_svc_rbf_moe, y_pred_rfc_moe, y_pred_abc_moe))
    y_pred_all = (stats.mode(y_all)[0][0]).astype(int)
    print 'MOE All Metrics: '
    all_moe_acc, all_moe_prec, all_moe_rec, all_moe_cm, all_moe_cm_full = get_scores(y_test,y_pred_all)
   

