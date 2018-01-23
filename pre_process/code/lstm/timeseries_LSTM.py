# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import your module here
from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.timeseries.python.timeseries import NumpyReader
from tensorflow.contrib.timeseries.python.timeseries.estimators import TimeSeriesRegressor
from train_lstm import _LSTMModel
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

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
def regression_valuate(y_pred,y_true):
    r2 = r2_score(y_true, y_pred)
    mse = mean_absolute_error(y_true, y_pred)
    return (mse,r2)

# main program here
if __name__ == '__main__':
    csv_file_name = '../../time_series/' + file_sets[0] + '_perf'+'.csv'
    reader = tf.contrib.timeseries.CSVReader(csv_file_name)
    data_row = pd.read_csv(csv_file_name)
    plt.plot(data_row.values[:,0], data_row.values[:,1])
    plt.savefig('../../plot/perfscores.png')
    data = {
            tf.contrib.timeseries.TrainEvalFeatures.TIMES: data_row.values[0:300,0],
            tf.contrib.timeseries.TrainEvalFeatures.VALUES: data_row.values[0:300,1],
            }
    reader = NumpyReader(data)
    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=16, window_size=40)
    estimator = TimeSeriesRegressor(
            model=_LSTMModel(num_features=1, num_units=128),
            optimizer=tf.train.AdamOptimizer(0.001))
    estimator.train(input_fn=train_input_fn, steps=2000)
    evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
    evaluation = estimator.evaluate(input_fn=evaluation_input_fn, steps=1)
    # Predict starting after the evaluation
    (predictions,) = tuple(estimator.predict(
            input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
                    evaluation, steps=50)))
    invaluate_result = regression_valuate(predictions['mean'].reshape(-1),data_row.values[300:350,1])
    plt.figure(figsize=(25, 5))
    plt.plot(data['times'].reshape(-1), data['values'].reshape(-1), label='origin')
    plt.plot(evaluation['times'].reshape(-1), evaluation['mean'].reshape(-1), label='evaluation')
    plt.plot(predictions['times'].reshape(-1), predictions['mean'].reshape(-1), label='prediction')
    plt.plot(data_row.values[300:350,0].reshape(-1), data_row.values[300:350,1].reshape(-1), label='groudtruth')
    plt.xlabel('time_step')
    plt.ylabel('values')
    plt.legend(loc=4)
    plt.title("lstm predict result", fontsize = 20)
    plt.savefig('../../plot/perf_predict_result_lstm.png')
    print(invaluate_result)         # (1.7127156066894531, 0.012876979251902898)

    """
    csv_file_name = '../time_series/' + file_sets[0] + '_perf'+'.csv'
    reader = tf.contrib.timeseries.CSVReader(csv_file_name)
    data = pd.read_csv(csv_file_name)
    plt.plot(data.values[:,0], data.values[:,1])
    plt.savefig('../plot/perfscores.png')
    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=2, window_size=10)
    with tf.Session() as sess:
        batch_data = train_input_fn.create_batch()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        one_batch = sess.run(batch_data[0])
        coord.request_stop()
    print('one_batch_data:', one_batch)
    """
    
    """
    x = np.array(range(1000))
    noise = np.random.uniform(-0.2, 0.2, 1000)
    y = np.sin(np.pi * x / 100) + x / 200. + noise
    plt.plot(x, y)
    plt.savefig('../plot/timeseries_sin_y.png')
    data = {
            tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
            tf.contrib.timeseries.TrainEvalFeatures.VALUES: y,
            }
    reader = NumpyReader(data)
    with tf.Session() as sess:
        full_data = reader.read_full()
        # 调用read_full方法会生成读取队列
        # 要用tf.train.start_queue_runners启动队列才能正常进行读取
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print(sess.run(full_data))
        coord.request_stop()
    train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(reader, batch_size=2, window_size=10)
    with tf.Session() as sess:
        batch_data = train_input_fn.create_batch()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        one_batch = sess.run(batch_data[0])
        coord.request_stop()
    print('one_batch_data:', one_batch)
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    