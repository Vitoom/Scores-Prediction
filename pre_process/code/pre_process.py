# -*- coding: utf-8 -*-

"""
Spyder Editor

This is a temporary script file.
"""

# import your module here

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
import matplotlib
from matplotlib.ticker import FuncFormatter
from scipy.stats import pearsonr
from skrebate import ReliefF
from collections import Counter
import seaborn as sbn
import os
import pickle as pkl

# (global) variable definition here

file_sets = [
        'DBID(1002089510)_INSTID(1)','DBID(2897570545)_INSTID(1)',
        'DBID(1227435885)_INSTID(1)','DBID(2949199900)_INSTID(1)',
        'DBID(1227435885)_INSTID(2)','DBID(3065831173)_INSTID(1)',
        'DBID(1254139675)_INSTID(1)','DBID(3111200895)_INSTID(1)',
        'DBID(1384807946)_INSTID(1)','DBID(3172835364)_INSTID(1)',
        'DBID(1624869053)_INSTID(1)','DBID(3204204681)_INSTID(1)',
        'DBID(1636599671)_INSTID(1)','DBID(3482311182)_INSTID(1)',
        'DBID(1636599671)_INSTID(2)','DBID(349165204)_INSTID(1)',
        'DBID(172908691)_INSTID(1)','DBID(3671658776)_INSTID(1)',
        'DBID(1855232979)_INSTID(1)','DBID(3671658776)_INSTID(2)',
        'DBID(1982696497)_INSTID(1)','DBID(3775482706)_INSTID(1)',
        'DBID(2031853600)_INSTID(1)','DBID(3775482706)_INSTID(2)',
        'DBID(2052255707)_INSTID(1)','DBID(4213264717)_INSTID(1)',
        'DBID(2238741707)_INSTID(1)','DBID(4215505906)_INSTID(1)',
        'DBID(2238741707)_INSTID(2)','DBID(4225426100)_INSTID(1)',
        'DBID(2328880794)_INSTID(1)','DBID(4291669003)_INSTID(1)',
        'DBID(2413621137)_INSTID(1)','DBID(4291669003)_INSTID(2)',
        'DBID(2612437783)_INSTID(1)','DBID(447326245)_INSTID(1)',
        'DBID(2644427317)_INSTID(1)','DBID(468957624)_INSTID(1)',
        'DBID(2707003786)_INSTID(1)','DBID(505574722)_INSTID(1)',
        'DBID(2762567375)_INSTID(1)','DBID(522516877)_INSTID(1)',
        'DBID(2768077198)_INSTID(1)','DBID(770699067)_INSTID(1)',
        'DBID(2778659381)_INSTID(1)','DBID(929227073)_INSTID(1)',
        'DBID(2778659381)_INSTID(2)','DBID(942093433)_INSTID(1)',
        'DBID(2802676787)_INSTID(1)','DBID(998852395)_INSTID(1)'
        ]

file_one = ['DBID(1624869053)_INSTID(1)','DBID(3204204681)_INSTID(1)',]

file_all = ["all_data"]

select_load = ['2080020','2080021','2080022','2080023','2080024','2080025','2080026',\
                  '2080027','2080028','2080029','2080030','2080031','2080032','2080033','2080034']

select_perf = ['2080040','2080041','2080042','2080043','2080044','2080045','2080046',\
               '2080047','2080048','2080049','2080050','2080051','2080052','2080053',\
               '2080054','2080055','2080056','2080057','2080058','2080059','2080060',\
                                       '2080061','2080062','2080063','2080064','2080065'] 

feature = ['2080020', '2080021','2080022', '2080023', '2080024', '2080025', 
           '2080026', '2080027','2080028', '2080029', '2080030', '2080031', 
           '2080032', '2080033','2080034', '2080040', '2080041', '2080042', 
           '2080043', '2080044','2080045', '2080046', '2080047', '2080048', 
           '2080049', '2080050','2080051', '2080052', '2080053', '2080054', 
           '2080055', '2080056','2080057', '2080058', '2080059', '2080060', 
           '2080061', '2080062','2080063', '2080064', '2080065',]

global db

global denominator

# class definition here

# function definition here

# plot percent hist plot
def to_percent(y, position):
    global denominator
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str("%.2f"%(100 * y/denominator))

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

# pearson index calculate
def pearson(X,y):
    scores = []
    pvalues = []
    for i in range(X.shape[1]):
        score, pvalue = pearsonr(X[:,i],y)
        scores += [score,]
        pvalues += [pvalue,]
    return scores,pvalues
    
# feature selection
def feature_selection(method,instance_db,target,percentage):
    instance_db_values = instance_db.values
    if method == "lasso":
        lassocv = LassoCV(max_iter=5000, normalize=True,alphas = [0.0001])
        lassocv.fit(instance_db_values,target)
        
        
        # sort features according to coef_
        coef = abs(lassocv.coef_)
        feature = zip(instance_db.columns, coef)
        feature = sorted(feature, key = lambda tup : tup[1], reverse = True)
        features = pd.DataFrame([tup[0] for tup in feature])
        
        # the features been chosen
        mask = pd.DataFrame([item[1] for item in feature]) >= feature[int(len(feature)*percentage)-1][1]
        feature_selected = pd.DataFrame(features)[:][np.asarray(mask).reshape(-1)]
        feature_selected = [item for items in feature_selected.values for item in items]
    
    if method == "pearson":
        scores, _ = pearson(instance_db_values,target)
        scores = np.nan_to_num(scores)
        scores = [abs(term) for term in scores]
        
        # sort features according to cores_ 
        feature = zip(instance_db.columns, scores)
        feature = sorted(feature, key = lambda tup : tup[1], reverse = True)
        features = pd.DataFrame([tup[0] for tup in feature])
        
        # the features been chosen
        mask = pd.DataFrame([item[1] for item in feature]) >= feature[int(len(feature)*percentage-1)][1]  
        feature_selected = pd.DataFrame(features)[:][np.asarray(mask).reshape(-1)]
        feature_selected = [item for items in feature_selected.values for item in items]
    
    if method == "reliefF":
        select = ReliefF(n_neighbors=5)
        instance_db_values = np.array(instance_db_values, np.float64)
        max_batch_instance = 5000
        if instance_db_values.shape[0] > max_batch_instance:
            for i in range(int(instance_db_values.shape[0]/max_batch_instance)):
                instance = instance_db_values[i * max_batch_instance : (i + 1) * max_batch_instance]
                tar = target[i * max_batch_instance : (i + 1) * max_batch_instance]
                select.fit_transform(instance,tar)
                if i == 0:
                    feature_impotances = select.feature_importances_
                else:
                    feature_impotances += np.asarray(select.feature_importances_)
            feature_impotances = list(feature_impotances)
        else:
            select.fit_transform(instance_db_values,target)
            feature_impotances = list(select.feature_importances_)
        # sort features according to importance_
        feature = zip(instance_db.columns, feature_impotances)
        feature = sorted(feature, key = lambda tup : tup[1], reverse = True)
        features = pd.DataFrame([tup[0] for tup in feature])
        
        # the features been chosen
        mask = pd.DataFrame([item[1] for item in feature]) >= feature[int(len(feature)*percentage-1)][1]   
        feature_selected = pd.DataFrame(features)[:][np.asarray(mask).reshape(-1)]
        feature_selected = [item for items in feature_selected.values for item in items]
        
    return feature_selected

# regression 
def regression(training_data, scores):
    # train sets and test sets split
    X_train, X_test, y_train, y_test = train_test_split(training_data, scores, test_size=0.2)
    
    # linear regression
    linear = linear_model.LinearRegression()
    linear.fit(X_train,pd.DataFrame(y_train).astype(float))
    linear_result = list(linear.predict(X_test))
    
    # Support Vector Regression
    svr = svm.SVR(kernel = 'linear')
    svr.fit(X_train,pd.DataFrame(y_train).astype(float))
    svr_result = list(svr.predict(X_test))
    
    # Multi-layer Perceptron
    mlp = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(10,10,10,10), max_iter=300)
    mlp.fit(X_train,pd.DataFrame(y_train).astype(float))
    mlp_result = list(mlp.predict(X_test))
  
    return linear_result, svr_result, mlp_result, list(y_test)

# classification
def classification(training_data, levels):
    # train sets and test sets split
    while True:
        X_train, X_test, y_train, y_test = train_test_split(training_data, levels, test_size=0.2)
        if(len(set(y_train)) > 1):
            break;
    
    # support vector classification
    _svm = svm.SVC(kernel = 'linear')
    _svm.fit(X_train,pd.DataFrame(y_train).astype(int))  
    svm_result = list(_svm.predict(X_test))
    
    # logistic regression
    logistic = LogisticRegression()
    logistic.fit(X_train,pd.DataFrame(y_train).astype(int))
    log_result = list(logistic.predict(X_test))
    
    # Extremely Randomized Trees
    rf = RandomForestClassifier(n_estimators = 10, max_features = 'log2')
    rf.fit(X_train,pd.DataFrame(y_train).astype(int).values)
    rf_result = list(rf.predict(X_test))
    
    # Multi-layer Perceptron classifier
    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10,10,10,10), max_iter=300)
    mlp.fit(X_train,pd.DataFrame(y_train).astype(int))
    mlp_result = list(mlp.predict(X_test))
    
    return svm_result, log_result, rf_result, mlp_result, list(y_test)

# plot histograph
def plot_hist(vector,size,xlabel,ylabel,_bins,filename,_rwidth,method="normal"):
    if method == "normal":
        f, ax = plt.subplots(figsize=size)
        f.tight_layout(pad=3)
        ax.hist(vector,bins=_bins, label=xlabel, color='b', alpha=.5, rwidth = _rwidth)
        ax.set_title(ylabel)
        ax.legend(loc='best')
        f.show()
        f.savefig("../plot/" + filename)
    if method == "percentage":
        f, ax = plt.subplots(figsize=size)
        f.tight_layout(pad=3)
        ax.hist(vector,bins=_bins, label=xlabel, color='b', alpha=.5, rwidth = _rwidth)
        ax.set_title(ylabel)
        # Create the formatter using the function to_percent. This multiplies all the
        # default labels by 100, making them all percentages
        formatter = FuncFormatter(to_percent)
        # Set the formatter
        f.gca().yaxis.set_major_formatter(formatter)
        ax.legend(loc='best')
        f.show()
        f.savefig("../plot/" + filename)
        
# to_csv
def to_csv(table,filename):
    table = pd.DataFrame(table)
    table.to_csv(path_or_buf='../csv/'+filename+'.csv')

# regeression valuate
def regression_valuate(y_pred,y_true):
    r2 = r2_score(y_true, y_pred)
    mse = mean_absolute_error(y_true, y_pred)
    return (mse,r2)

# main method
def run_method(feature_percentage, run_turn):
    global db
    
    scores_load_aggre = []
    levels_load_aggre = []
    scores_perf_aggre = []
    levels_perf_aggre = []
    
    
    y_test_regr_predict_load = {"linearR_un":[],
                                "linearR_lasso":[],
                                "linearR_pearson":[],
                                "linearR_reliefF":[],
                                "svr_un":[],
                                "svr_lasso":[],
                                "svr_pearson":[],
                                "svr_reliefF":[],
                                "mlp_un":[],
                                "mlp_lasso":[],
                                "mlp_pearson":[],
                                "mlp_reliefF":[],
                                }
    y_test_regr_load = {"None":[],
                       "lasso":[],
                       "pearson":[],
                       "reliefF":[],
                        }
    
    y_test_cla_predict_load = {"svm_un":[],
                               "svm_lasso":[],
                               "svm_pearson":[],
                               "svm_reliefF":[],
                               "logisticR_un":[],
                               "logisticR_lasso":[],
                               "logisticR_pearson":[],
                               "logisticR_reliefF":[],
                               "rf_un":[],
                               "rf_lasso":[],
                               "rf_pearson":[],
                               "rf_reliefF":[],
                               "mlpC_un":[],
                               "mlpC_lasso":[],
                               "mlpC_pearson":[],
                               "mlpC_reliefF":[],
                               }
    y_test_cla_load = {"None":[],
                       "lasso":[],
                       "pearson":[],
                       "reliefF":[],
                        }
    y_test_regr_predict_perf = {"linearR_un":[],
                                "linearR_lasso":[],
                                "linearR_pearson":[],
                                "linearR_reliefF":[],
                                "svr_un":[],
                                "svr_lasso":[],
                                "svr_pearson":[],
                                "svr_reliefF":[],
                                "mlp_un":[],
                                "mlp_lasso":[],
                                "mlp_pearson":[],
                                "mlp_reliefF":[],
                                }
    y_test_regr_perf = {"None":[],
                       "lasso":[],
                       "pearson":[],
                       "reliefF":[],
                        }
    
    y_test_cla_predict_perf = {"svm_un":[],
                               "svm_lasso":[],
                               "svm_pearson":[],
                               "svm_reliefF":[],
                               "logisticR_un":[],
                               "logisticR_lasso":[],
                               "logisticR_pearson":[],
                               "logisticR_reliefF":[],
                               "rf_un":[],
                               "rf_lasso":[],
                               "rf_pearson":[],
                               "rf_reliefF":[],
                               "mlpC_un":[],
                               "mlpC_lasso":[],
                               "mlpC_pearson":[],
                               "mlpC_reliefF":[],
                               }
    y_test_cla_perf = {"None":[],
                       "lasso":[],
                       "pearson":[],
                       "reliefF":[],
                        }

    feature_select_load = {"lasso":{},
                           "pearson":{},
                           "reliefF":{},
                           }
    feature_select_perf = {"lasso":{},
                           "pearson":{},
                           "reliefF":{},
                           }
    
    sets = file_all  # file_sets 
    
    for turn,file in enumerate(sets):
        db = pd.read_csv('../csv/' + file + '.csv')
        print('open file ','../csv/' + file + '.csv')
        
        # Preprocessing
        # Imputation of missing values
        db_values = db.values
        instance_db = db_values[:, 4:-4]
        for i in range(instance_db.shape[0]):
            for j in range(instance_db.shape[1]):
                # process the null
                if instance_db[i][j] == 'null':
                    # assign NaN to null for future process
                    instance_db[i][j] = 'NaN'
        # tackle column whose members are all NaN
        for j in range(instance_db.shape[1]):
            if list(instance_db[:,j]) == list(['NaN']*instance_db.shape[0]):
                instance_db[:,j] = [0]*instance_db.shape[0]
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0,verbose=1)
        imp.fit(instance_db)
        instance_db = imp.transform(instance_db)
        # Normalization
        scaler = StandardScaler()
        scaler.fit(instance_db)
        instance_db = scaler.transform(instance_db)
        db_values[:, 4:-4] = instance_db
        db = pd.DataFrame(db_values,columns = db.columns)
        
        # target 
        target_db = db_values[:, -4:]
        
        # load scores as target
        scores_load = target_db[:, 0]
        
        # aggregate load scores
        scores_load_aggre = scores_load_aggre + list(scores_load)
        
        # load levels as target
        levels_load = target_db[:,1]
        
        # aggregate load levels
        levels_load_aggre = levels_load_aggre + list(levels_load)
               
        # convert 'A'... to 0... 
        levels_load = [ord(x) - ord('A') for x in levels_load] 
        
        # invalidate feature selection
        # instance_data = db[select_load]
        instance_data = db[feature]
        
        # regression process
        linear_result, svr_result, mlp_result, y_test = regression(instance_data, scores_load)
        y_test_regr_predict_load["linearR_un"] += linear_result
        y_test_regr_predict_load["svr_un"] += svr_result
        y_test_regr_predict_load["mlp_un"] += mlp_result
        y_test_regr_load["None"] += y_test
        
        # classification process
        count = Counter(levels_load)
        balance = np.array([count[i] for i in count])/len(levels_load) > 0.1
        _select = len(set(balance)) > 1
        if len(set(levels_load)) > 1 and _select:
            svm_result, log_result, rf_result, mlp_result, y_test = classification(instance_data, levels_load)
            y_test_cla_predict_load["svm_un"] += svm_result
            y_test_cla_predict_load["logisticR_un"] += log_result
            y_test_cla_predict_load["rf_un"] += rf_result
            y_test_cla_predict_load["mlpC_un"] += mlp_result
            y_test_cla_load["None"] += y_test
    
        # load scores lasso
        print('load scores lasso')
        instance_data = db[feature]
        # save select features
        feature_select_load["lasso"][turn] = feature_selection("lasso", instance_data,scores_load, percentage=feature_percentage)
        instance_data = db[feature_select_load["lasso"][turn]]
        
        # regression process
        linear_result, svr_result, mlp_result, y_test = regression(instance_data, scores_load)
        y_test_regr_predict_load["linearR_lasso"] += linear_result
        y_test_regr_predict_load["svr_lasso"] += svr_result
        y_test_regr_predict_load["mlp_lasso"] += mlp_result
        y_test_regr_load["lasso"] += y_test
        
        # classification process
        count = Counter(levels_load)
        balance = np.array([count[i] for i in count])/len(levels_load) > 0.1
        _select = len(set(balance)) > 1
        if len(set(levels_load)) > 1 and _select:
            svm_result, log_result, rf_result, mlp_result, y_test = classification(instance_data, levels_load)
            y_test_cla_predict_load["svm_lasso"] += svm_result
            y_test_cla_predict_load["logisticR_lasso"] += log_result
            y_test_cla_predict_load["rf_lasso"] += rf_result
            y_test_cla_predict_load["mlpC_lasso"] += mlp_result
            y_test_cla_load["lasso"] += y_test
        
        # load scores pearson
        print('load scores pearson')
        instance_data = db[feature]
        # save select features
        feature_select_load["pearson"][turn] = feature_selection("pearson",instance_data,scores_load, percentage=feature_percentage)
        instance_data = db[feature_select_load["pearson"][turn]]
        
        # regression process
        linear_result, svr_result, mlp_result, y_test = regression(instance_data, scores_load)
        y_test_regr_predict_load["linearR_pearson"] += linear_result
        y_test_regr_predict_load["svr_pearson"] += svr_result
        y_test_regr_predict_load["mlp_pearson"] += mlp_result
        y_test_regr_load["pearson"] += y_test
        
        # classification process
        count = Counter(levels_load)
        balance = np.array([count[i] for i in count])/len(levels_load) > 0.1
        _select = len(set(balance)) > 1
        if len(set(levels_load)) > 1 and _select:
            svm_result, log_result, rf_result, mlp_result, y_test = classification(instance_data, levels_load)
            y_test_cla_predict_load["svm_pearson"] += svm_result
            y_test_cla_predict_load["logisticR_pearson"] += log_result
            y_test_cla_predict_load["rf_pearson"] += rf_result
            y_test_cla_predict_load["mlpC_pearson"] += mlp_result
            y_test_cla_load["pearson"] += y_test
        
        # load levels reliefF
        print('load levels reliefF')
        instance_data = db[feature]
        # save select features
        feature_select_load["reliefF"][turn] = feature_selection("reliefF",instance_data,np.array(levels_load,np.int32), percentage=feature_percentage)
        instance_data = db[feature_select_load["reliefF"][turn]]
        
        # regression process
        linear_result, svr_result, mlp_result, y_test = regression(instance_data, scores_load)
        y_test_regr_predict_load["linearR_reliefF"] += linear_result
        y_test_regr_predict_load["svr_reliefF"] += svr_result
        y_test_regr_predict_load["mlp_reliefF"] += mlp_result
        y_test_regr_load["reliefF"] += y_test
        
        # classification processcount = Counter(levels_load)
        balance = np.array([count[i] for i in count])/len(levels_load) > 0.1
        _select = len(set(balance)) > 1
        if len(set(levels_load)) > 1 and _select:
            svm_result, log_result, rf_result, mlp_result, y_test = classification(instance_data, levels_load)
            y_test_cla_predict_load["svm_reliefF"] += svm_result
            y_test_cla_predict_load["logisticR_reliefF"] += log_result
            y_test_cla_predict_load["rf_reliefF"] += rf_result
            y_test_cla_predict_load["mlpC_reliefF"] += mlp_result
            y_test_cla_load["reliefF"] += y_test
        
        # performance scores as target
        scores_perf = target_db[:, 2]
        
        # aggregate load scores
        scores_perf_aggre = scores_perf_aggre + list(scores_perf)
        
        # performance levels as target
        levels_perf = target_db[:,3]
        
        # aggregate load levels
        levels_perf_aggre = levels_perf_aggre + list(levels_perf)
        
        # convert 'A'... to 0... 
        levels_perf = [ord(x) - ord('A') for x in levels_perf]
        
        # invalidate feature selection
        # instance_data = db[select_perf]
        instance_data = db[feature]
        
        # regression process
        linear_result, svr_result, mlp_result, y_test = regression(instance_data, scores_perf)
        y_test_regr_predict_perf["linearR_un"] += linear_result
        y_test_regr_predict_perf["svr_un"] += svr_result
        y_test_regr_predict_perf["mlp_un"] += mlp_result
        y_test_regr_perf["None"] += y_test
        
        # classification process
        count = Counter(levels_perf)
        balance = np.array([count[i] for i in count])/len(levels_load) > 0.1
        _select = len(set(balance)) > 1
        if len(set(levels_perf)) > 1 and _select:
            svm_result, log_result, rf_result, mlp_result, y_test = classification(instance_data, levels_perf)
            y_test_cla_predict_perf["svm_un"] += svm_result
            y_test_cla_predict_perf["logisticR_un"] += log_result
            y_test_cla_predict_perf["rf_un"] += rf_result
            y_test_cla_predict_perf["mlpC_un"] += mlp_result
            y_test_cla_perf["None"] += y_test
        
        # performance scores lasso
        print('performance scores lasso')
        instance_data = db[feature]
        # save select features
        feature_select_perf["lasso"][turn]= feature_selection("lasso",instance_data,scores_perf, percentage=feature_percentage)
        instance_data = db[feature_select_perf["lasso"][turn]]
        
        # regression process
        linear_result, svr_result, mlp_result, y_test = regression(instance_data, scores_perf)
        y_test_regr_predict_perf["linearR_lasso"] += linear_result
        y_test_regr_predict_perf["svr_lasso"] += svr_result
        y_test_regr_predict_perf["mlp_lasso"] += mlp_result
        y_test_regr_perf["lasso"] += y_test
        
        # classification process
        count = Counter(levels_perf)
        balance = np.array([count[i] for i in count])/len(levels_load) > 0.1
        _select = len(set(balance)) > 1
        if len(set(levels_perf)) > 1 and _select:
            svm_result, log_result, rf_result, mlp_result, y_test = classification(instance_data, levels_perf)
            y_test_cla_predict_perf["svm_lasso"] += svm_result
            y_test_cla_predict_perf["logisticR_lasso"] += log_result
            y_test_cla_predict_perf["rf_lasso"] += rf_result
            y_test_cla_predict_perf["mlpC_lasso"] += mlp_result
            y_test_cla_perf["lasso"] += y_test
        
        # performance scores pearson
        print('performance scores pearson')
        instance_data = db[feature]
        # save select features
        feature_select_perf["pearson"][turn]= feature_selection("pearson", instance_data, scores_perf, percentage=feature_percentage)
        instance_data = db[feature_select_perf["pearson"][turn]]
        
        # regression process
        linear_result, svr_result, mlp_result, y_test = regression(instance_data, scores_perf)
        y_test_regr_predict_perf["linearR_pearson"] += linear_result
        y_test_regr_predict_perf["svr_pearson"] += svr_result
        y_test_regr_predict_perf["mlp_pearson"] += mlp_result
        y_test_regr_perf["pearson"] += y_test
        
        # classification process
        count = Counter(levels_perf)
        balance = np.array([count[i] for i in count])/len(levels_load) > 0.1
        _select = len(set(balance)) > 1
        if len(set(levels_perf)) > 1 and _select:
            svm_result, log_result, rf_result, mlp_result, y_test = classification(instance_data, levels_perf)
            y_test_cla_predict_perf["svm_pearson"] += svm_result
            y_test_cla_predict_perf["logisticR_pearson"] += log_result
            y_test_cla_predict_perf["rf_pearson"] += rf_result
            y_test_cla_predict_perf["mlpC_pearson"] += mlp_result
            y_test_cla_perf["pearson"] += y_test
    
        # performance levels reliefF
        print('performance levels reliefF')
        instance_data = db[feature]
        # save select features
        feature_select_perf["reliefF"][turn] = feature_selection("reliefF", instance_data, np.array(levels_perf, np.int32), percentage=feature_percentage)
        instance_data = db[feature_select_perf["reliefF"][turn]]
        
        # regression process
        linear_result, svr_result, mlp_result, y_test = regression(instance_data, scores_perf)
        y_test_regr_predict_perf["linearR_reliefF"] += linear_result
        y_test_regr_predict_perf["svr_reliefF"] += svr_result
        y_test_regr_predict_perf["mlp_reliefF"] += mlp_result
        y_test_regr_perf["reliefF"] += y_test
        
        # classification process
        count = Counter(levels_perf)
        balance = np.array([count[i] for i in count])/len(levels_load) > 0.1
        _select = len(set(balance)) > 1
        if len(set(levels_perf)) > 1 and _select:
            svm_result, log_result, rf_result, mlp_result, y_test = classification(instance_data, levels_perf)
            y_test_cla_predict_perf["svm_reliefF"] += svm_result
            y_test_cla_predict_perf["logisticR_reliefF"] += log_result
            y_test_cla_predict_perf["rf_reliefF"] += rf_result
            y_test_cla_predict_perf["mlpC_reliefF"] += mlp_result
            y_test_cla_perf["reliefF"] += y_test
        
    # common feature selected
    save_path = "../pkl/select_feature/"+"select_feature-" + "f_percentage-" + str(feature_percentage) + "-run_turn-" + str(run_turn) + ".pkl"
    output = open(save_path, 'wb') 
    dump_data = (feature_select_load, feature_select_perf)
    pkl.dump(dump_data, output)
    output.close()
    """
    set_num = len(sets)
    sets = [set(feature_select_load["lasso"][i]) for i in range(set_num)]
    feature_select_load["lasso"][set_num] = set.intersection(*sets)
    sets = [set(feature_select_perf["lasso"][i]) for i in range(set_num)]
    feature_select_perf["lasso"][set_num] = set.intersection(*sets)
    sets = [set(feature_select_load["pearson"][i]) for i in range(set_num)]
    feature_select_load["pearson"][set_num] = set.intersection(*sets)
    sets = [set(feature_select_perf["pearson"][i]) for i in range(set_num)]
    feature_select_perf["pearson"][set_num] = set.intersection(*sets)
    sets = [set(feature_select_load["reliefF"][i]) for i in range(set_num)]
    feature_select_load["reliefF"][set_num] = set.intersection(*sets)
    sets = [set(feature_select_perf["reliefF"][i]) for i in range(set_num)]
    feature_select_perf["reliefF"][set_num] = set.intersection(*sets)
    """
    
    # plot aggregate summary
    """
    denominator = 53522
    plot_hist(scores_load_aggre, (7, 5), "load score", "# of load score", 20,"histograph_of_load_scores",0.65,"percentage")
    plot_hist(levels_load_aggre, (5, 5), "load level", "# of load level",4,"histograph_of_load_levels",0.7,"percentage")
    plot_hist(scores_perf_aggre, (7, 5), "performance score", "# of performance score", 20, "histograph_of_perf_scores",\
                                                                                                          0.65,"percentage")
    plot_hist(levels_perf_aggre, (5, 5), "performance level", "# of performance level",4, "histograph_of_perf_levels",\
                                                                                                          0.7,"percentage")
    plot_hist([item for items in (pd.DataFrame(y_test_regr_predict_perf["mlp_lasso"])-pd.DataFrame(\
                                  y_test_regr_perf["lasso"])).values for item in items], (7, 5),\
                              "residual", "# of residual(mlp)", 20, "histograph_of_mlp_residuals",0.65)
    plot_hist([item for items in (pd.DataFrame(y_test_regr_predict_perf["svr_lasso"])-pd.DataFrame(\
                                  y_test_regr_perf["lasso"])).values for item in items], (7, 5),\
                              "residual", "# of residual(svr)", 20, "histograph_of_svr_residuals",0.65)
    """
    
    # valuate results
    load_regr_valuate = {"feature selection":["None","lasso","pearson","reliefF"],
                        "linearR":[],
                        "svr":[],
                        "mlp":[]
                        }
    load_regr_valuate["linearR"].append(regression_valuate(\
                    y_test_regr_predict_load["linearR_un"],y_test_regr_load["None"]))
    load_regr_valuate["svr"].append(regression_valuate(\
                    y_test_regr_predict_load["svr_un"],y_test_regr_load["None"]))
    load_regr_valuate["mlp"].append(regression_valuate(\
                    y_test_regr_predict_load["mlp_un"],y_test_regr_load["None"]))
    load_regr_valuate["linearR"].append(regression_valuate(\
                    y_test_regr_predict_load["linearR_lasso"],y_test_regr_load["lasso"]))
    load_regr_valuate["svr"].append(regression_valuate(\
                    y_test_regr_predict_load["svr_lasso"],y_test_regr_load["lasso"]))
    load_regr_valuate["mlp"].append(regression_valuate(\
                    y_test_regr_predict_load["mlp_lasso"],y_test_regr_load["lasso"]))
    load_regr_valuate["linearR"].append(regression_valuate(\
                    y_test_regr_predict_load["linearR_pearson"],y_test_regr_load["pearson"]))
    load_regr_valuate["svr"].append(regression_valuate(\
                    y_test_regr_predict_load["svr_pearson"],y_test_regr_load["pearson"]))
    load_regr_valuate["mlp"].append(regression_valuate(\
                    y_test_regr_predict_load["mlp_pearson"],y_test_regr_load["pearson"]))
    load_regr_valuate["linearR"].append(regression_valuate(\
                    y_test_regr_predict_load["linearR_reliefF"],y_test_regr_load["reliefF"]))
    load_regr_valuate["svr"].append(regression_valuate(\
                    y_test_regr_predict_load["svr_reliefF"],y_test_regr_load["reliefF"]))
    load_regr_valuate["mlp"].append(regression_valuate(\
                    y_test_regr_predict_load["mlp_reliefF"],y_test_regr_load["reliefF"]))
    print("load regression evaluation")
    print(load_regr_valuate)
    
    load_cla_valuate = {"feature selection":["None","lasso","pearson","reliefF"],
                        "svm":[],
                        "logisticR":[],
                        "rf":[],
                        "mlpC":[]
                        }
    load_cla_valuate["svm"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["svm_un"],y_test_cla_load["None"],average = 'macro')[:3])
    load_cla_valuate["logisticR"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["logisticR_un"],y_test_cla_load["None"],average = 'macro')[:3])
    load_cla_valuate["rf"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["rf_un"],y_test_cla_load["None"],average = 'macro')[:3])
    load_cla_valuate["mlpC"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["mlpC_un"],y_test_cla_load["None"],average = 'macro')[:3])
    load_cla_valuate["logisticR"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["logisticR_lasso"],y_test_cla_load["lasso"],average = 'macro')[:3])
    load_cla_valuate["svm"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["svm_lasso"],y_test_cla_load["lasso"],average = 'macro')[:3])
    load_cla_valuate["rf"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["rf_lasso"],y_test_cla_load["lasso"],average = 'macro')[:3])
    load_cla_valuate["mlpC"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["mlpC_lasso"],y_test_cla_load["lasso"],average = 'macro')[:3])
    load_cla_valuate["svm"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["svm_pearson"],y_test_cla_load["pearson"],average = 'macro')[:3])
    load_cla_valuate["logisticR"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["logisticR_pearson"],y_test_cla_load["pearson"],average = 'macro')[:3])
    load_cla_valuate["rf"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["rf_pearson"],y_test_cla_load["pearson"],average = 'macro')[:3])
    load_cla_valuate["mlpC"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["mlpC_pearson"],y_test_cla_load["pearson"],average = 'macro')[:3])
    load_cla_valuate["svm"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["svm_reliefF"],y_test_cla_load["reliefF"],average = 'macro')[:3])
    load_cla_valuate["logisticR"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["logisticR_reliefF"],y_test_cla_load["reliefF"],average = 'macro')[:3])
    load_cla_valuate["rf"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["rf_reliefF"],y_test_cla_load["reliefF"],average = 'macro')[:3])
    load_cla_valuate["mlpC"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_load["mlpC_reliefF"],y_test_cla_load["reliefF"],average = 'macro')[:3])
    print("load classification evaluation")
    print(load_cla_valuate)
    
    perf_regr_valuate = {"feature selection":["None","lasso","pearson","reliefF"],
                        "linearR":[],
                        "svr":[],
                        "mlp":[]
                        }
    perf_regr_valuate["linearR"].append(regression_valuate(\
                    y_test_regr_predict_perf["linearR_un"],y_test_regr_perf["None"]))
    perf_regr_valuate["svr"].append(regression_valuate(\
                    y_test_regr_predict_perf["svr_un"],y_test_regr_perf["None"]))
    perf_regr_valuate["mlp"].append(regression_valuate(\
                    y_test_regr_predict_perf["mlp_un"],y_test_regr_perf["None"]))
    perf_regr_valuate["linearR"].append(regression_valuate(\
                    y_test_regr_predict_perf["linearR_lasso"],y_test_regr_perf["lasso"]))
    perf_regr_valuate["svr"].append(regression_valuate(\
                    y_test_regr_predict_perf["svr_lasso"],y_test_regr_perf["lasso"]))
    perf_regr_valuate["mlp"].append(regression_valuate(\
                   y_test_regr_predict_perf["mlp_lasso"],y_test_regr_perf["lasso"]))
    perf_regr_valuate["linearR"].append(regression_valuate(\
                    y_test_regr_predict_perf["linearR_pearson"],y_test_regr_perf["pearson"]))
    perf_regr_valuate["svr"].append(regression_valuate(\
                    y_test_regr_predict_perf["svr_pearson"],y_test_regr_perf["pearson"]))
    perf_regr_valuate["mlp"].append(regression_valuate(\
                   y_test_regr_predict_perf["mlp_pearson"],y_test_regr_perf["pearson"]))
    perf_regr_valuate["linearR"].append(regression_valuate(\
                    y_test_regr_predict_perf["linearR_reliefF"],y_test_regr_perf["reliefF"]))
    perf_regr_valuate["svr"].append(regression_valuate(\
                    y_test_regr_predict_perf["svr_reliefF"],y_test_regr_perf["reliefF"]))
    perf_regr_valuate["mlp"].append(regression_valuate(\
                   y_test_regr_predict_perf["mlp_reliefF"],y_test_regr_perf["reliefF"]))
    print("performance regression evaluation")
    print(perf_regr_valuate)
    
    perf_cla_valuate = {"feature selection":["None","lasso","pearson","reliefF",],
                        "svm":[],
                        "logisticR":[],
                        "rf":[],
                        "mlpC":[]
                        }
    perf_cla_valuate["svm"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["svm_un"],y_test_cla_perf["None"],average = 'macro')[:3])
    perf_cla_valuate["logisticR"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["logisticR_un"],y_test_cla_perf["None"],average = 'macro')[:3])
    perf_cla_valuate["rf"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["rf_un"],y_test_cla_perf["None"],average = 'macro')[:3])
    perf_cla_valuate["mlpC"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["mlpC_un"],y_test_cla_perf["None"],average = 'macro')[:3])
    perf_cla_valuate["svm"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["svm_lasso"],y_test_cla_perf["lasso"],average = 'macro')[:3])
    perf_cla_valuate["logisticR"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["logisticR_lasso"],y_test_cla_perf["lasso"],average = 'macro')[:3])
    perf_cla_valuate["rf"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["rf_lasso"],y_test_cla_perf["lasso"],average = 'macro')[:3])
    perf_cla_valuate["mlpC"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["mlpC_lasso"],y_test_cla_perf["lasso"],average = 'macro')[:3])
    perf_cla_valuate["svm"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["svm_pearson"],y_test_cla_perf["pearson"],average = 'macro')[:3])
    perf_cla_valuate["logisticR"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["logisticR_pearson"],y_test_cla_perf["pearson"],average = 'macro')[:3])
    perf_cla_valuate["rf"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["rf_pearson"],y_test_cla_perf["pearson"],average = 'macro')[:3])
    perf_cla_valuate["mlpC"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["mlpC_pearson"],y_test_cla_perf["pearson"],average = 'macro')[:3])
    perf_cla_valuate["logisticR"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["logisticR_reliefF"],y_test_cla_perf["reliefF"],average = 'macro')[:3])
    perf_cla_valuate["svm"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["svm_reliefF"],y_test_cla_perf["reliefF"],average = 'macro')[:3])
    perf_cla_valuate["rf"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["rf_reliefF"],y_test_cla_perf["reliefF"],average = 'macro')[:3])
    perf_cla_valuate["mlpC"].append(precision_recall_fscore_support(\
                    y_test_cla_predict_perf["mlpC_reliefF"],y_test_cla_perf["reliefF"],average = 'macro')[:3])
    print("performance classification evaluation")
    print(perf_cla_valuate)
    
    return load_regr_valuate, load_cla_valuate, perf_regr_valuate, perf_cla_valuate

# main program here
if  __name__ == '__main__':
    method_r = ["linearR", "svr", "mlp"]
    method_c = ["svm", "logisticR", "rf", "mlpC"]
    feature_sel = ["None", "lasso", "pearson", "reliefF"]
    evaluation_r = ["MSE", "R2",]
    evaluation_c = ["precision", "recall", "fscore"]
    d_r = ["feature_percentage", "regr_method", "feature_select", "evaluation", "value",]
    d_c = ["feature_percentage", "cla_method", "feature_select",  "evaluation", "value",]
    percentage = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #  
    
    pkl_path = "../pkl/Dataframe_save.pkl"
    if not os.path.exists(pkl_path):
        df_load_regr = pd.DataFrame(columns=d_r)
        df_load_cla = pd.DataFrame(columns=d_c)
        df_perf_regr = pd.DataFrame(columns=d_r)
        df_perf_cla = pd.DataFrame(columns=d_c)
        
        for percen in percentage:
            run_turn = 6
            while run_turn:
                run_turn -= 1
                load_regr_valuate, load_cla_valuate, perf_regr_valuate, perf_cla_valuate = run_method(percen, run_turn)
                for m in method_r:
                    for f in range(len(feature_sel)):
                        for e in range(len(evaluation_r)):
                            df_load_regr.loc[df_load_regr.shape[0]] = [percen, m, feature_sel[f], evaluation_r[e],\
                                             load_regr_valuate[m][f][e],]
                            df_perf_regr.loc[df_perf_regr.shape[0]] = [percen, m, feature_sel[f], evaluation_r[e],\
                                             perf_regr_valuate[m][f][e],]
                for m1 in method_c:
                    for f1 in range(len(feature_sel)):
                        for e1 in range(len(evaluation_c)):
                            df_load_cla.loc[df_load_cla.shape[0]] = [percen, m1, feature_sel[f1], evaluation_c[e1],\
                                             load_cla_valuate[m1][f1][e1],]
                            df_perf_cla.loc[df_perf_cla.shape[0]] = [percen, m1, feature_sel[f1], evaluation_c[e1],\
                                             perf_cla_valuate[m1][f1][e1],]
        
        df_load_regr_un = df_load_regr[df_load_regr["feature_select"].isin(["None"])]
        df_load_cla_un = df_load_cla[df_load_cla["feature_select"].isin(["None"])]
        df_perf_regr_un = df_perf_regr[df_perf_regr["feature_select"].isin(["None"])]
        df_perf_cla_un = df_perf_cla[df_perf_cla["feature_select"].isin(["None"])]
        
        output = open(pkl_path, 'wb')
        dump_data = (df_load_regr, df_load_cla, df_perf_regr, df_perf_cla,\
                     df_load_regr_un, df_load_cla_un, df_perf_regr_un, df_perf_cla_un,)
        pkl.dump(dump_data, output)
        output.close()
    else:
        output = open(pkl_path, 'rb')
        df_load_regr, df_load_cla, df_perf_regr, df_perf_cla, df_load_regr_un, df_load_cla_un, df_perf_regr_un, df_perf_cla_un,\
           = pkl.load(output)
        output.close()
    
    # tendency plot                  
    plt.figure(figsize=(8,35))
    sbn.set_style("whitegrid")
    sbn.factorplot(x="feature_percentage", y="value", hue="regr_method", col="evaluation", ci=None, legend=True, data=df_load_regr, kind="point", dodge=True, size=8, aspect=2)
    plt.tight_layout(pad=3)
    sbn.set(font_scale=2)
    plt.savefig("../plot/result/tendency/" + "load_regression_evaluate")
    
    plt.figure(figsize=(8,35))
    sbn.set_style("whitegrid")
    sbn.factorplot(x="feature_percentage", y="value", hue="cla_method", col="evaluation", ci=None, legend=True, data=df_load_cla, kind="point", dodge=True, size=8, aspect=2)
    plt.tight_layout(pad=3)
    sbn.set(font_scale=1.7)
    plt.savefig("../plot/result/tendency/" + "load_classification_evaluate")
    
    plt.figure(figsize=(8,35))
    sbn.set_style("whitegrid")
    sbn.factorplot(x="feature_percentage", y="value", hue="regr_method", col="evaluation", ci=None, legend=True, data=df_perf_regr, kind="point", dodge=True, size=8, aspect=2)
    plt.tight_layout(pad=3)
    sbn.set(font_scale=2)
    plt.savefig("../plot/result/tendency/" + "perf_regression_evaluate")
    
    plt.figure(figsize=(8,35))
    sbn.set_style("whitegrid")
    sbn.factorplot(x="feature_percentage", y="value", hue="cla_method", col="evaluation", ci=None, legend=True, data=df_perf_cla, kind="point", dodge=True, size=8, aspect=2)
    plt.tight_layout(pad=3)
    sbn.set(font_scale=2)
    plt.savefig("../plot/result/tendency/" + "perf_classification_evaluate")
    
    # histogram plot
    plt.figure(figsize=(8,6))
    sbn.set_style("whitegrid")
    sbn.factorplot(x="evaluation", y="value", hue="regr_method", kind="bar", legend=False, size=8, data=df_load_regr_un)
    plt.tight_layout(pad=3)
    plt.legend(loc=0)
    sbn.set(font_scale=2)
    plt.savefig("../plot/result/histogram/" + "load_regression_evaluate")
    
    plt.figure(figsize=(8,6))
    plt.tight_layout(pad=3)
    sbn.set(font_scale=1.7)
    sbn.set_style("whitegrid")
    sbn.factorplot(x="evaluation", y="value", hue="cla_method", kind="bar", legend=False, size=8, data=df_load_cla_un)
    plt.legend(loc=0)
    plt.yticks(np.arange(0, 1.5, 0.1), fontsize=14)
    plt.savefig("../plot/result/histogram/" + "load_classification_evaluate")
    
    plt.figure(figsize=(8,6))
    plt.tight_layout(pad=3)
    sbn.set(font_scale=1.7)
    sbn.set_style("whitegrid")
    sbn.factorplot(x="evaluation", y="value", hue="regr_method", kind="bar", legend=False, size=8, data=df_load_regr_un)
    plt.legend(loc=0)
    plt.savefig("../plot/result/histogram/" + "perf_regression_evaluate")
    
    plt.figure(figsize=(8,6))
    plt.tight_layout(pad=3)
    sbn.set(font_scale=1.7)
    sbn.set_style("whitegrid")
    sbn.factorplot(x="evaluation", y="value", hue="cla_method", kind="bar", legend=False, size=8, data=df_perf_cla_un)
    plt.legend(loc=0)
    plt.yticks(np.arange(0, 1.5, 0.1), fontsize=14)
    plt.savefig("../plot/result/histogram/" + "perf_classification_evaluate")
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    