#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 19:09:20 2018

@author: vito
"""

# import your module here

import numpy as np
np.random.seed(12345)
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
from time import time
from scipy import stats
import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs

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
        'DBID(2802676787)_INSTID(1)','DBID(998852395)_INSTID(1)',
        ]

file_sets = [
        'DBID(1002089510)_INSTID(1)','DBID(2897570545)_INSTID(1)',
        'DBID(1227435885)_INSTID(1)','DBID(2949199900)_INSTID(1)',
        'DBID(1227435885)_INSTID(2)','DBID(3065831173)_INSTID(1)',
        'DBID(1254139675)_INSTID(1)','DBID(3111200895)_INSTID(1)',
        'DBID(1384807946)_INSTID(1)','DBID(3172835364)_INSTID(1)',
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
        'DBID(2802676787)_INSTID(1)','DBID(998852395)_INSTID(1)',
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

# pearson index calculate
def pearson(X,y):
    scores = []
    pvalues = []
    for i in range(X.shape[1]):
        score, pvalue = pearsonr(X[:,i],y)
        scores += [score,]
        pvalues += [pvalue,]
    return scores, pvalues

# feature selection
def feature_selection(method,instance_db,target,percentage):
    instance_db_values = instance_db.values
    if method == "lasso":
        lassocv = LassoCV(max_iter=5000, normalize=True, alphas = [0.0001])
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
def regression(training_data, scores, parameter):
    # train sets and test sets split
    X_train, X_test, y_train, y_test = train_test_split(training_data, scores, test_size=0.2)
    
    # Multi-layer Perceptron
    mlp = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=(10,)*parameter, max_iter=500)
    mlp.fit(X_train,pd.DataFrame(y_train).astype(float))
    mlp_result = list(mlp.predict(X_test))
    
    return mlp_result, list(y_test)
    
# classification
def classification_rf(training_data, levels, parameter):
    # train sets and test sets split
    while True:
        X_train, X_test, y_train, y_test = train_test_split(training_data, levels, test_size=0.2)
        if(len(set(y_train)) > 1):
            break;
    
    # Extremely Randomized Trees
    rf = RandomForestClassifier(n_estimators=parameter, max_features='log2')
    rf.fit(X_train, pd.DataFrame(y_train).astype(int).values)
    rf_result = list(rf.predict(X_test))
    
    return rf_result, list(y_test)
    
def classification_mlpc(training_data, levels, parameter):
    # train sets and test sets split
    while True:
        X_train, X_test, y_train, y_test = train_test_split(training_data, levels, test_size=0.2)
        if(len(set(y_train)) > 1):
            break;
    
    # Multi-layer Perceptron classifier
    mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10,)*parameter, max_iter=1000)
    mlp.fit(X_train, pd.DataFrame(y_train).astype(int).values)
    mlp_result = list(mlp.predict(X_test))
    
    return mlp_result, list(y_test)

def regression_valuate(y_pred,y_true):
    r2 = r2_score(y_true, y_pred)
    mse = mean_absolute_error(y_true, y_pred)
    return (mse,r2)

def run_method():
    global db
    
    sets = file_all  # file_sets 
    
    mlp_para = list(range(2,21))
    rf_para = list(range(2,21))
    
    columns_cla = ["method", "parameter(# of trees or # of layers)", "evaluation", "value"]
    columns_regr = ["method", "parameter(# of layers)", "evaluation", "value"]
    df_load_cla = pd.DataFrame(columns=columns_cla)
    df_load_regr = pd.DataFrame(columns=columns_regr)
    df_perf_cla = pd.DataFrame(columns=columns_cla)
    df_perf_regr = pd.DataFrame(columns=columns_regr)
    
    for turn,file in enumerate(sets):
        db = pd.read_csv('../csv/' + file + '.csv')
        print('open file ','../csv/' + file + '.csv')
        
        # Preprocessingm
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
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=1)
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
        
        # load levels as target
        levels_load = target_db[:, 1]
        
        # convert 'A'... to 0... 
        levels_load = [ord(x) - ord('A') for x in levels_load] 
        
        # invalidate feature selection
        # instance_data = db[select_load]
        instance_data = db[feature]
        
        # regression process
        for i in range(6):
            for para in mlp_para:
                mlp_result, y_test = regression(instance_data, scores_load, para)
                _mse, _R2 = regression_valuate(mlp_result, y_test)
                df_load_regr.loc[df_load_regr.shape[0]] = ["mlp", para, "mse", _mse]
                df_load_regr.loc[df_load_regr.shape[0]] = ["mlp", para, "R2", _R2]
        
        # classification process
        count = Counter(levels_load)
        balance = np.array([count[i] for i in count])/len(levels_load) > 0.1
        _select = len(set(balance)) > 1
        if len(set(levels_load)) > 1 and _select:
            for i in range(6):
                for para in rf_para:
                    rf_result, y_test = classification_rf(instance_data, levels_load, para)
                    p, r, f = precision_recall_fscore_support(y_test, rf_result, average = 'macro')[:3]
                    df_load_cla.loc[df_load_cla.shape[0]] = ["rf", para, "precision", p]
                    df_load_cla.loc[df_load_cla.shape[0]] = ["rf", para, "recall", r]
                    df_load_cla.loc[df_load_cla.shape[0]] = ["rf", para, "fscore", f]
                for para in mlp_para:
                    mlp_result, y_test = classification_mlpc(instance_data, levels_load, para)
                    p, r, f = precision_recall_fscore_support(y_test, mlp_result, average = 'macro')[:3]
                    df_load_cla.loc[df_load_cla.shape[0]] = ["mlpc", para, "precision", p]
                    df_load_cla.loc[df_load_cla.shape[0]] = ["mlpc", para, "recall", r]
                    df_load_cla.loc[df_load_cla.shape[0]] = ["mlpc", para, "fscore", f]
        
        # performance scores as target
        scores_perf = target_db[:, 2]
        
        # performance levels as target
        levels_perf = target_db[:,3]
        
        # convert 'A'... to 0... 
        levels_perf = [ord(x) - ord('A') for x in levels_perf]
        
        # invalidate feature selection
        # instance_data = db[select_perf]
        instance_data = db[feature]
        
        # regression process
        for i in range(6):
            for para in mlp_para:
                mlp_result, y_test = regression(instance_data, scores_perf, para)
                _mse, _R2 = regression_valuate(mlp_result, y_test)
                df_perf_regr.loc[df_perf_regr.shape[0]] = ["mlp", para, "mse", _mse]
                df_perf_regr.loc[df_perf_regr.shape[0]] = ["mlp", para, "R2", _R2]
        
        # classification processhape[0]
        count = Counter(levels_perf)
        balance = np.array([count[i] for i in count])/len(levels_load) > 0.1
        _select = len(set(balance)) > 1
        if len(set(levels_perf)) > 1 and _select:
            for i in range(6):
                for para in rf_para:
                    rf_result, y_test = classification_rf(instance_data, levels_perf, para)
                    p, r, f = precision_recall_fscore_support(y_test, rf_result, average = 'macro')[:3]
                    df_perf_cla.loc[df_perf_cla.shape[0]] = ["rf", para, "precision", p]
                    df_perf_cla.loc[df_perf_cla.shape[0]] = ["rf", para, "recall", r]
                    df_perf_cla.loc[df_perf_cla.shape[0]] = ["rf", para, "fscore", f]
                for para in mlp_para:
                    mlp_result, y_test = classification_mlpc(instance_data, levels_perf, para)
                    p, r, f = precision_recall_fscore_support(y_test, mlp_result, average = 'macro')[:3]
                    df_perf_cla.loc[df_perf_cla.shape[0]] = ["mlpc", para, "precision", p]
                    df_perf_cla.loc[df_perf_cla.shape[0]] = ["mlpc", para, "recall", r]
                    df_perf_cla.loc[df_perf_cla.shape[0]] = ["mlpc", para, "fscore", f]
    return df_load_cla, df_load_regr, df_perf_cla, df_perf_regr

if __name__ == "__main__":
    pkl_path = "../pkl/parameter_select/parametr_select_load.pkl"
    if not os.path.exists(pkl_path):
        df_load_cla, df_load_regr, df_perf_cla, df_perf_regr = run_method()
        output = open(pkl_path, 'wb')
        dump_data = (df_load_cla, df_load_regr, df_perf_cla, df_perf_regr)
        pkl.dump(dump_data, output)
        output.close()
    else:
        output = open(pkl_path, 'rb')
        df_load_cla, df_load_regr, df_perf_cla, df_perf_regr = pkl.load(output)
        output.close()
    
    plt.figure(figsize=(8,35))
    sbn.set_style("whitegrid")
    sbn.factorplot(x="parameter(# of layers)", y="value", hue="evaluation", col="method", ci=None, legend=True, data=df_load_regr, kind="point", dodge=True, size=8, aspect=2)
    plt.tight_layout(pad=3)
    sbn.set(font_scale=5)
    plt.savefig("../plot/parameter_select/" + "load_regr_mlp.png")
    
    plt.figure(figsize=(8,35))
    sbn.set_style("whitegrid")
    sbn.factorplot(x="parameter(# of trees or # of layers)", y="value", hue="evaluation", ci=None, legend=True, data=df_load_cla[df_load_cla["method"].isin(["rf"])], kind="point", dodge=True, size=8, aspect=2)
    plt.tight_layout(pad=3)
    sbn.set(font_scale=5)
    plt.savefig("../plot/parameter_select/" + "load_cla_rf.png")
    
    plt.figure(figsize=(8,35))
    sbn.set_style("whitegrid")
    sbn.factorplot(x="parameter(# of trees or # of layers)", y="value", hue="evaluation", ci=None, legend=True, data=df_load_cla[df_load_cla["method"].isin(["mlpc"])], kind="point", dodge=True, size=8, aspect=2)
    plt.tight_layout(pad=3)
    sbn.set(font_scale=5)
    plt.savefig("../plot/parameter_select/" + "load_cla_mlpc.png")
    
    plt.figure(figsize=(8,35))
    sbn.set_style("whitegrid")
    sbn.factorplot(x="parameter(# of layers)", y="value", hue="evaluation", col="method", ci=None, legend=True, data=df_perf_regr, kind="point", dodge=True, size=8, aspect=2)
    plt.tight_layout(pad=3)
    sbn.set(font_scale=5)
    plt.savefig("../plot/parameter_select/" + "perf_regr_mlp.png")
    
    plt.figure(figsize=(8,35))
    sbn.set_style("whitegrid")
    sbn.factorplot(x="parameter(# of trees or # of layers)", y="value", hue="evaluation", ci=None, legend=True, data=df_perf_cla[df_perf_cla["method"].isin(["rf"])], kind="point", dodge=True, size=8, aspect=2)
    plt.tight_layout(pad=3)
    sbn.set(font_scale=5)
    plt.savefig("../plot/parameter_select/" + "perf_cla_rf.png")    
    
    plt.figure(figsize=(8,35))
    sbn.set_style("whitegrid")
    sbn.factorplot(x="parameter(# of trees or # of layers)", y="value", hue="evaluation", ci=None, legend=True, data=df_perf_cla[df_perf_cla["method"].isin(["mlpc"])], kind="point", dodge=True, size=8, aspect=2)
    plt.tight_layout(pad=3)
    sbn.set(font_scale=5)
    plt.savefig("../plot/parameter_select/" + "perf_cla_mlpc.png")










