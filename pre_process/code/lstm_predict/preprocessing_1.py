#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 21:02:11 2018

@author: dingwangxiang
"""

# import your module here
import numpy as np
import pandas as pd
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Dropout
# from sklearn.preprocessing import MinMaxScaler
# from trainingset_selection import TrainingSetSelection
from keras.models import load_model, save_model
from keras.utils import plot_model
#from keras.layers.core import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
from preprocessing import get_ids_and_files_in_dir, percentile_remove_outlier, MinMaxScaler, NormalDistributionScaler, binning_date_y
import os
from matplotlib import pyplot as plt

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
        'DBID(2802676787)_INSTID(1)','DBID(998852395)_INSTID(1)',
        ]

file_one = ['DBID(1002089510)_INSTID(1)',]

file_all = ["all_data"]

select_load = ['2080020','2080021','2080022','2080023','2080024','2080025','2080026',\
                  '2080027','2080028','2080029','2080030','2080031','2080032','2080033','2080034']

select_perf = ['2080040','2080041','2080042','2080043','2080044','2080045','2080046',\
               '2080047','2080048','2080049','2080050','2080051','2080052','2080053',\
               '2080054','2080055','2080056','2080057','2080058','2080059','2080060',\
                                       '2080061','2080062','2080063','2080064','2080065']

columns = ['Unnamed: 0', 'SnapId', 'StartTime', 'EndTime', '2080020', '2080021',
           '2080022', '2080023', '2080024', '2080025', '2080026', '2080027',
           '2080028', '2080029', '2080030', '2080031', '2080032', '2080033',
           '2080034', '2080040', '2080041', '2080042', '2080043', '2080044',
           '2080045', '2080046', '2080047', '2080048', '2080049', '2080050',
           '2080051', '2080052', '2080053', '2080054', '2080055', '2080056',
           '2080057', '2080058', '2080059', '2080060', '2080061', '2080062',
           '2080063', '2080064', '2080065', 'LoadScore', 'LoadLevel', 'PerfScore',
           'PerfLevel']

select_load_columns = ['2080020', '2080021', '2080022', '2080023', '2080024', '2080025', 
                       '2080026', '2080027', '2080028', '2080029', '2080030', '2080031', 
                       '2080032', '2080033', '2080034', '2080040', '2080041', '2080042', 
                       '2080043', '2080044', '2080045', '2080046', '2080047', '2080048', 
                       '2080049', '2080050', '2080051', '2080052', '2080053', '2080054', 
                       '2080055', '2080056', '2080057', '2080058', '2080059', '2080060', 
                       '2080061', '2080062', '2080063', '2080064', '2080065', 'LoadScore', 
                       'PerfScore',
                       ]

select_load_columns_one = ['LoadScore',]

select_perf_columns = ['2080020', '2080021', '2080022', '2080023', '2080024', '2080025', 
                       '2080026', '2080027', '2080028', '2080029', '2080030', '2080031', 
                       '2080032', '2080033', '2080034', '2080040', '2080041', '2080042', 
                       '2080043', '2080044', '2080045', '2080046', '2080047', '2080048', 
                       '2080049', '2080050', '2080051', '2080052', '2080053', '2080054', 
                       '2080055', '2080056', '2080057', '2080058', '2080059', '2080060', 
                       '2080061', '2080062', '2080063', '2080064', '2080065', 'LoadScore', 
                       'PerfScore',
                       ]

select_perf_columns_one = ['PerfScore',]

num_episodes = 600

# class definition here

class EpisodeHistory:
    def __init__(self,
                 capacity,
                 title = "...",
                 ylabel = "...",
                 xlabel="...",
                 verbose = True,
                 plot_episode_count=num_episodes,
                 max_value_per_episode=1.2,
                 num_plot=1,
                 label=["0","1","2","3"]):

        self.datas = {}
        for i in range(num_plot):
            self.datas[i] = np.zeros(capacity, dtype=float)
        
        self.plot_episode_count = plot_episode_count
        self.max_value_per_episode = max_value_per_episode
        self.label = label
        self.num_plot = num_plot 
        self.point_plot = {}
        self.fig = None
        self.ax = None
        self.verbose = verbose
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        
        self.fig, self.ax = plt.subplots(figsize=(14, 7), facecolor='w', edgecolor='k')
        self.fig.canvas.set_window_title(self.title)
        
        self.ax.set_xlim(0, self.plot_episode_count + 5)
        self.ax.set_ylim(0, self.max_value_per_episode)
        self.ax.yaxis.grid(True)
        
        self.ax.set_title(self.title, fontsize=22)
        self.ax.set_xlabel(self.xlabel, fontsize=22)
        self.ax.set_ylabel(self.ylabel, fontsize=22)
        
        color_set = ['b', 'g', 'r', 'c']
        for i in range(num_plot):
            self.point_plot[i] = plt.plot([], [], linewidth=2.0, c=color_set[i], label=self.label[i])

    def __getitem__(self, index):
        return self.datas[index]

    def __setitem__(self, index, value):
        self.datas[index] = value

    def update_plot(self, episode_index):
        plot_right_edge = episode_index
        plot_left_edge = max(0, plot_right_edge - self.plot_episode_count)

        # Update point plot.
        for i in range(self.num_plot):
            x = range(plot_left_edge, plot_right_edge)
            y = self.datas[i][plot_left_edge:plot_right_edge]
            self.point_plot[i][0].set_xdata(x)
            self.point_plot[i][0].set_ydata(y)
            self.ax.set_xlim(plot_left_edge, plot_left_edge + self.plot_episode_count)

        # Repaint the surface.
        plt.legend(fontsize=22)
        plt.draw()
        plt.pause(0.0001)
        
        
class NeuralNetwork():
    def __init__(self,
                 model_save_dir,
                 output_dir=".",
                 model_file_prefix='model',
                 training_set_id_range=(0, np.Inf),
                 training_set_length=3,
                 scaler = 'mm',
                 **kwargs):
        """
        :param training_set_dir: directory contains the training set files. File format: 76.csv
        :param model_save_dir: directory to receive trained model and model weights. File format: model-76.json/model-weight-76.h5
        :param model_file_prefix='model': file        print ("Model evaluation: {}".format(score))                  # [0.29132906793909186, 0.91639871695672837] prefix for model file
        :param training_set_range=(0, np.Inf): enterprise ids in this range (a, b) would be analyzed. PS: a must be less than b
        :param training_set_length=3: first kth columns in training set file will be used as training set and the following one is expected value
        :param train_test_ratio=3: the ratio of training set size to test set size when splitting input data
        :param output_dir=".": output directory for prediction files
        :param scaler: scale data set using - mm: MinMaxScaler, norm: NormalDistributionScaler
        :param **kwargs: lstm_output_dim=4: output dimension of LSTM layer;
                        activation_lstm='relu': activation function for LSTM layers;
                        activation_dense='relu': activation function for Dense layer;
                        activation_last='softmax': activation function for last layer;
                        drop_out=0.2: fraction of input units to drop;
                        np_epoch=25, the number of epoches to train the model. epoch is one forward pass and one backward pass of all the training examples;
                        batch_size=100: number of samples per gradient update. The higher the batch size, the more memory space you'll need;
                        loss='categorical_crossentropy': loss function;
                        optimizer='rmsprop'
        """
        
        self.model_save_dir = model_save_dir
        self.model_file_prefix = model_file_prefix
        self.training_set_id_range = training_set_id_range
        self.training_set_length = training_set_length
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.scaler = scaler
        self.test_size = kwargs.get('test_size', 0.2)
        self.lstm_output_dim = kwargs.get('lstm_output_dim', 8)
        self.activation_lstm = kwargs.get('activation_lstm', 'relu')
        self.activation_dense = kwargs.get('activation_dense', 'relu')
        self.activation_last = kwargs.get('activation_last', 'softmax')    # softmax for multiple output
        self.dense_layer = kwargs.get('dense_layer', 2)  # at least 2 layers
        self.lstm_layer = kwargs.get('lstm_layer', 2) # at least 2 layers
        self.drop_out = kwargs.get('drop_out', 0.2)
        self.nb_epoch = kwargs.get('nb_epoch', num_episodes)
        self.batch_size = kwargs.get('batch_size', 10000)
        self.loss = kwargs.get('loss', 'categorical_crossentropy')
        self.optimizer = kwargs.get('optimizer', 'rmsprop')


    def NN_model_train(self, trainX, trainY, testX, testY, model_save_path, end, lookback):
        """
        :param trainX: training data set
        :param trainY: expect value of training data
        :param testX: test data segt
        :param testY: expect value of test data
        :param model_save_path: h5 file to store the trained model
        :param override: override existing models
        :return: model after training
        """
        input_dim = trainX[0].shape[1]
        output_dim = trainY.shape[1]
        # print predefined parameters of current model:
        model = Sequential()
        # applying a LSTM layer with x dim output and y dim input. Use dropout parameter to avoid overfit
        model.add(LSTM(output_dim=self.lstm_output_dim,
                       input_dim=input_dim,
                       activation=self.activation_lstm,
                       dropout=self.drop_out,
                       return_sequences=True))
        for i in range(self.lstm_layer-2):
            model.add(LSTM(output_dim=self.lstm_output_dim,
                       activation=self.activation_lstm,
                       dropout=self.drop_out,
                       return_sequences=True ))
        # return sequences should be False to avoid dim error when concatenating with dense layer
        model.add(LSTM(output_dim=self.lstm_output_dim, activation=self.activation_lstm, dropout_U=self.drop_out))
        # applying a full connected NN to accept output from LSTM layer
        for i in range(self.dense_layer-1):
            model.add(Dense(output_dim=self.lstm_output_dim, activation=self.activation_dense))
            model.add(Dropout(self.drop_out))
        model.add(Dense(output_dim=output_dim, activation=self.activation_last))
        # configure the learning process
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        # train the model with fixed number of epoches
        h = model.fit(x=trainX, y=trainY, nb_epoch=self.nb_epoch, batch_size=self.batch_size, validation_data=(testX, testY))
        
        num_plot = 4
        
        trace = EpisodeHistory(num_episodes,
                               num_plot=num_plot,
                               title = "Training hisotry plot",
                               ylabel = "accuracy or loss",
                               xlabel= "Training epochs",
                               label=list(h.history.keys())
                               )
        for i, key in enumerate(h.history.keys()):
            trace[i][:] = h.history[key]
        trace.update_plot(num_episodes)
        trace.fig.savefig(fname="../../plot/training_history_plot/lookback-" + str(lookback) +"-lstm-traning-history.png")
        
        if end:
            model.summary()
            plot_model(model, to_file='model.png')
        score = model.evaluate(trainX, trainY, self.batch_size)
        # store model to json file
        save_model(model, model_save_path)
        return score


    @staticmethod
    def NN_prediction(dataset, model_save_path):
        dataset = np.asarray(dataset)
        if not os.path.exists(model_save_path):
            raise ValueError("Lstm model not found! Train one first or check your input path: {}".format(model_save_path))
        model = load_model(model_save_path)
        predict_class = model.predict_classes(dataset)
        class_prob = model.predict_proba(dataset)
        return predict_class, class_prob


    def model_train_predict_test(self, dataX, dataY, end,  lookback, override=False,):      
        # remove outlier records
        """
        df_selected = percentile_remove_outlier(df_selected, filter_start=0, filter_end=1+self.training_set_length)
        print(df_selected)
        """
        # scale the train columns
        print ("Scaling...")
        if self.scaler == 'mm':
            copy = dataX[:,:,dataX.shape[2]-1]
            dataX[:,:,dataX.shape[2]-1], minVal, maxVal, _bin_boundary = MinMaxScaler(copy)
            for i in range(dataX.shape[2]-1):
                copy = dataX[:,:,i]
                dataX[:,:,i], null1, null2, null3 =  MinMaxScaler(copy)
        elif self.scaler == 'norm':
            pass
            # target_collection, meanVal, stdVal = NormalDistributionScaler(target_collection, start_col_index=0, end_col_index=self.training_set_length)
        else:
            raise ValueError("Argument scaler must be mm or norm!")
        # bin date y
        bin_boundary = [0,50,75,90]
        dataY, bin_boundary = binning_date_y(dataY, y_col=self.training_set_length, n_group=5, bin_boundary=bin_boundary)
        print ("Bin boundary is {}".format(bin_boundary))
        # get train and test dataset
        print ("Randomly selecting training set and test set...")
        # convert y label to one-hot dummy label
        if len(set(dataY)) == 1:
            return (1010,[1010,1010])
        y_dummy_label = np.asarray(pd.get_dummies(dataY))
        # format train, test, validation data
        count = 0
        while True:
            x_sub, x_test, y_sub, y_test = train_test_split(dataX, y_dummy_label, test_size=self.test_size)
            x_train, x_val, y_train, y_val = train_test_split(x_sub, y_sub, test_size=self.test_size)
            if count == 10:
                return (1010,[1010,1010])
            def to_list(x):
                to_list = []
                for i in range(x.shape[0]):
                    result = 0
                    for j in range(x.shape[1]):
                        result += x[i][j] * (j + 1)
                    to_list.append(result)
                return to_list
            if len(set(to_list(y_train))) > 1 :
                break;
        # create and fit the NN model
        model_save_path = self.model_save_dir + "/" + self.model_file_prefix + ".h5"
        # check if model file exists
        if not os.path.exists(model_save_path) or override:
            score = self.NN_model_train(x_train, y_train, x_val, y_val, model_save_path=model_save_path, end=end, lookback=lookback)
            print ("Models and their parameters are stored in {}".format(model_save_path))
        else:
            score = [1010, 1010]
        # generate prediction for training
        print ("Predicting the output of validation set...")           
        val_predict_class, val_predict_prob = self.NN_prediction(x_test, model_save_path=model_save_path)
        # statistic of discrepancy between expected value and real value
        total_sample_count = len(val_predict_class)
        val_test_label = np.asarray([list(x).index(1) for x in y_test])
        match_count = (np.asarray(val_predict_class) == np.asarray(val_test_label.ravel())).sum()
        return float(match_count) / total_sample_count, score

# function definition here

# look back dataset
def create_interval_dataset(dataset, lookback_num):
    """
    :param dataset: input array of time intervals + '-'
    :param look_back: each training set feature length
    :return: convert an array of values into a dataset matrix. create_interval_dataset
    """
    origin_instance_num = dataset.shape[0]
    instance_num = dataset.shape[0] - lookback_num
    feature_num = dataset.shape[1]
    dataX = np.zeros( (instance_num, lookback_num+1, feature_num) )  # weather or not contain the instance now
    dataY = np.zeros(instance_num)
    for i in range(lookback_num, origin_instance_num):
        for j in range(lookback_num+1):              # weather or not contain the instance now
            dataX[i-lookback_num][j][:] = dataset[i-lookback_num+j][:]
        dataY[i-lookback_num] = dataset[i][-1]
    return dataX, dataY

# Imputation of missing values
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

def lstm_predict(dataX, dataY, end, lookback):
    output_dir = "./cluster_lstm_model"                                        # "/your_local_path/RNN_prediction_2/cluster_lstm_model"
    training_set_length = 50
    dense_layer = 2
    model_file_prefix = 'model-' + "lookback-" + str(lookback) 
    model_save_dir = output_dir + "/" + model_file_prefix
    obj_NN = NeuralNetwork(output_dir=output_dir,
                           model_save_dir=model_save_dir,
                           model_file_prefix=model_file_prefix,
                           training_set_length=training_set_length,
                           dense_layer=dense_layer)
    print ("Train NN model and test!")
    return obj_NN.model_train_predict_test(dataX, dataY, end=end, override=False, lookback=lookback)

# main program here
if __name__ == '__main__':
    sets = file_sets # file_one
    
    lookback_set = list(range(1, 30, 2))

    evaluate = np.zeros( (len(lookback_set), 3) )

    for loop,lookback in enumerate(lookback_set):
        
        for turn,file in enumerate(sets):
            db = pd.read_csv('../../csv/' + file + '.csv')
            print('open file ../../csv/' + file + '.csv')
            # Imputation of missing values
            db_values = db.values
            instance_db = db_values[:, 4 : -4]
            for i in range(instance_db.shape[0]):
                for j in range(instance_db.shape[1]):
                    # process the null
                    if instance_db[i][j] == 'null':
                        # assign NaN to null for future process
                        instance_db[i][j] = np.nan
            transform(instance_db)
            db_values[:, 4 : -4] = instance_db
            db = pd.DataFrame(db_values, columns = db.columns)
            
            # performance predicrt
            dataset_perf = db[select_perf_columns].values
            X_sub, Y_sub = create_interval_dataset(dataset_perf, lookback)
            if turn == 0:
                dataX, dataY = X_sub, Y_sub
            else:
                dataX = np.vstack( (dataX, X_sub) )
                dataY = np.append(dataY, Y_sub)
        end = True # if turn == len(sets) - 1 else False
        precision, score = lstm_predict(dataX, dataY, end=end, lookback=lookback)
        

        evaluate[loop,:] = score + [precision,]

        
        print ("Model evaluation [training loss, training prescision] : {}".format(score))
        print ("Precision using test dataset is {}".format(precision))
        
    num_plot = 3
    
    labels = ["tran_loss", "tran_acc", "acc"]
    
    trace = EpisodeHistory(len(lookback_set),
                           num_plot=num_plot,
                           title = "Test accuracy plot",
                           ylabel = "accuracy or loss",
                           xlabel= "lookback number",
                           label=labels
                           )
    for i in range(len(labels)):
        trace[i][:] = list(evaluate[:,i])
    trace.update_plot(len(lookback_set))
    trace.fig.savefig(fname="../../plot/training_select_lookback/history.png")
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        
        
        