#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 15:44:09 2017

@author: dingwangxiang
"""
import pandas as pd
from matplotlib import pyplot as plt

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

plt.figure(figsize=(15,45))

for turn,file in enumerate(file_sets):
    db = pd.read_csv('../csv/' + file + '.csv')
    print('open file ','../csv/' + file + '.csv')
    
    # Preprocessing
    # Imputation of missing values
    db_values = db.values
    
    # target 
    target_db = db_values[:, -4:]
    
    # load scores as target
    scores_load = target_db[:, 0]
    
    plt.subplot(17,3,turn+1)
    plt.plot(list(range(scores_load.shape[0])),scores_load)
     
plt.suptitle("Load Scores Diagram", fontsize=16,x=0.5,y=0.998)
plt.tight_layout(pad=2)
plt.savefig("../plot/load_diagram")
    
plt.figure(figsize=(15,45))

for turn,file in enumerate(file_sets):
    db = pd.read_csv('../csv/' + file + '.csv')
    print('open file ','../csv/' + file + '.csv')
    
    # Preprocessing
    # Imputation of missing values
    db_values = db.values
    
    # target 
    target_db = db_values[:, -4:]
    
    # performance scores as target
    scores_perf = target_db[:, 2]
    
    plt.subplot(17,3,turn+1)
    plt.plot(list(range(scores_load.shape[0])),scores_load)

plt.suptitle("Performance Scores Diagram", fontsize=16,x=0.5,y=0.998)
plt.tight_layout(pad=2)
plt.savefig("../plot/performance_diagram")

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    