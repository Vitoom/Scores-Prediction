#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 16:24:28 2017

@author: dingwangxiang
"""

# import your module here
import pandas as pd
import numpy as np

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


# class definition here

# function definition here

def transform(data):
    for i in range(data.shape[1]):
        if pd.isnull(data[:,i]).sum() == 0:
            continue
        elif pd.isnull(data[:,i]).sum() == data.shape[0]:
            data[:,i] = [0] * data.shape[0]
        else:
            valid_val = []
            valid = [int, float]
            for j in range(data.shape[0]):
                if type(data[j][i]) in valid and not np.isnan(float(data[j][i])):
                    valid_val.append(data[j][i])
            if len(valid_val) == 0:
                mean = 0
            else:
                mean = sum(valid_val)/len(valid_val)
            for j in range(data.shape[0]):
                if np.isnan(float(data[j][i])):
                    data[j][i] = mean
    return data

# main program here
if  __name__ == '__main__':
    control = True
    for turn,file in enumerate(file_sets):
        db = pd.read_csv('../csv/' + file + '.csv')
        print('open file ','../csv/' + file + '.csv')
        
        # Preprocessing
        # Imputation of missing values
        db_values = db.values
        instance_db = db_values[:, 4:-5]
        for i in range(instance_db.shape[0]):
            for j in range(instance_db.shape[1]):
                # process the null
                if instance_db[i][j] == 'null':
                    # assign NaN to null for future process
                    instance_db[i][j] = np.nan
        transform(instance_db)
        db_values[:, 4:-5] = instance_db
        db = pd.DataFrame(db_values,columns = db.columns)
        global container
        if control:
            container = db
            control = False
        else:
            container = pd.concat([container, db], ignore_index=False,)
    container = container.sort_values(['SnapId'])
    container.index = list(range(container.shape[0]))
    container.to_csv(path_or_buf = '../csv/'+'all_data'+'.csv', columns = container.columns[1:])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    