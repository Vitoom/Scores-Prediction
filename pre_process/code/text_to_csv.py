#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 12:28:29 2017

@author: dingwangxiang
"""

# import your module here
import pandas as pd

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

file_all = ["all_data"]

# class definition here

# function definition here

# main program here
if __name__ == '__main__':
    for file in file_sets:
        print(file)
        dic = {}
        f = open('../Awr_his_perf/'+file+'.txt','r+',encoding='utf-8')
        f.readline()
        example = f.readline()
        section = example.partition('LoadScore')
        prefix,empty,surfix = section[0].partition(' ')
        dic[prefix.split(':')[0]] = [prefix.split(':')[1],]
        front, mid, rear = surfix.partition(' EndTime')
        dic[front.partition(':')[0]] = [front.partition(':')[2],]
        rear = mid[1:] + rear
        dic[rear.partition(':')[0]] = [rear.partition(':')[2],]
        example = section[1] + section[2]
        terms = example.split()
        for term in terms:
            dic[term.split(':')[0]] = [term.split(':')[1],]
        example = f.readline()
        example = example[3:]
        terms = example.split()
        for term in terms:
            dic[term.split(':')[0]] = [term.split(':')[1],]
        while len(f.readline()) != 0 :
            example = f.readline()
            section = example.partition('LoadScore')
            prefix,empty,surfix = section[0].partition(' ')
            dic[prefix.split(':')[0]] = dic[prefix.split(':')[0]] + [prefix.split(':')[1],]
            front, mid, rear = surfix.partition(' EndTime')
            dic[front.partition(':')[0]] = dic[front.partition(':')[0]] + [front.partition(':')[2],]
            rear = mid[1:] + rear
            dic[rear.partition(':')[0]] = dic[rear.partition(':')[0]] + [rear.partition(':')[2],]
            example = section[1] + section[2]
            terms = example.split()
            for term in terms:
                dic[term.split(':')[0]] = dic[term.split(':')[0]] + [term.split(':')[1],]
            example = f.readline()
            example = example[3:]
            terms = example.split()
            for term in terms:
                dic[term.split(':')[0]] = dic[term.split(':')[0]] + [term.split(':')[1],]   
        table = pd.DataFrame(dic)
        cols = table.columns.tolist()
        cols = cols[-2:] + [cols[-7]] + cols[:-7] + [cols[-5],cols[-6],cols[-3],cols[-4]]
        table = table[cols]
        table.to_csv(path_or_buf='../csv/'+file+'.csv')