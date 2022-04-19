'''
Sample preparation scrip
'''

import glob
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def created_naive_forecast(dataset, window_size, prediction_step=1, target_index=0):
    '''
    Create naive forecast.
    '''
    dataset = np.array(dataset)
    y = []
    for i in range(0, len(dataset) - window_size - prediction_step + 1):
        begin = i
        end = i + window_size
        ####
        if len(dataset.shape) > 1:
            _y = dataset[end - 1: end + prediction_step - 1, target_index]
        else:
            _y = dataset[end - 1: end + prediction_step - 1]
        ####
        #_y = dataset[end - 1: end + prediction_step - 1, target_index]
        y.append(_y)
    return np.array(y)

def import_latest_data():
    '''
    Find latest csv file in data folder.
    '''
    list_of_files = glob.glob('data/*.csv') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    return pd.read_csv(latest_file, index_col=0)

def import_data(name):
    '''
    Find latest csv file in data folder.
    '''
    latest_file = 'data/'+name
    if os.path.exists(latest_file): # * means all if need specific format then *.csv
        print(latest_file)
        data = pd.read_csv(latest_file, index_col=0)
    else:
        print(latest_file + " does not exist.")
        data = []
    return data

def df_split(df, split):
    '''
    Split dataset based on split parameter.
    '''
    n = len(df)
    split_time = int(n*split)
    train_df = df[0:split_time]
    test_df = df[split_time:]
    # print("Train set shape is (%s, %s)" % (train_df.shape))
    # print("Test set shape is  (%s, %s)" % (test_df.shape))

    return train_df, test_df

def normalize_data(train_df, test_df):
    '''
    Normalize data
    '''
    train_mean = train_df.mean()
    train_std = train_df.std()

    scaler = {'mean': train_mean, 'std': train_std}

    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, test_df, scaler

def invert_normalize(scaler, y):
    '''
    Inverting normlization
    '''
    mean = scaler['mean'][0]
    std = scaler['std'][0]

    y_inverted = y*std + mean
    return y_inverted

def scale(train_df, test_df):
    '''
    Scale data between -1 and 1
    '''
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_df)
    # transform train
    #train_df = train_df.reshape(train_df.shape[0], train_df.shape[1])
    train_scaled = scaler.transform(train_df)
    # transform test
    #test_df = test_df.reshape(test_df.shape[0], test_df.shape[1])
    test_scaled = scaler.transform(test_df)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, y):
    
    array = np.concatenate([y, X], axis = 1)
    #array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[:, 0]

def create_windowed_1st_step(dataset, window_size, prediction_step, target_index):
    '''
    Create rolling window dataset.
    '''
    dataset = np.array(dataset)
    x = []
    y = []
    for i in range(0, len(dataset) - window_size - prediction_step + 1):
        begin = i
        end = i + window_size
        _x = dataset[begin:end]
        ####
        if len(dataset.shape) > 1:
            _y = dataset[end:end + prediction_step, target_index]
        else:
            _y = dataset[end:end + prediction_step]
        ####
        #_y = dataset[end:end + prediction_step, target_index]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)

def create_1st_step_y(quantiles, y_train, y_test):
    y_traink = y_train[:,]
    for k in range(len(quantiles)-1):
        y_traink = np.concatenate((y_traink, y_train[:,]), axis=1)
    #print(y_traink.shape)

    y_testk = y_test[:,]
    for k in range(len(quantiles)-1):
        y_testk = np.concatenate((y_testk, y_test[:,]), axis=1)
    #print(y_testk.shape)
    return y_traink, y_testk

def sample_creation(df, config, prediction_step = 1):
    '''
    Import, split, normalize data
    and return rolling window sample.
    '''
    quantiles = config['QUANTILES']
    split = config['SPLIT']
    window_size = config['WINDOW_SIZE']
    target_index = 0

    train_df, test_df = df_split(df, split)
    train_df, test_df, scaler = normalize_data(train_df, test_df)

    #scaler, train_df, test_df = scale(train_df, test_df)

    batch_size = train_df.shape[0] - 1 - window_size

    x_train, y_train = create_windowed_1st_step(train_df, window_size, prediction_step, target_index)
    x_test, y_test   = create_windowed_1st_step(test_df, window_size, prediction_step, target_index)

    y_traink, y_testk = create_1st_step_y(quantiles, y_train, y_test)

    data = {'TRAIN_DF': train_df,
            'TEST_DF': test_df,
            'X_TRAIN': x_train,
            'Y_TRAIN': y_train,
            'X_TEST': x_test,
            'Y_TEST': y_test,
            'Y_TRAINK': y_traink,
            'Y_TESTK': y_testk,
            'BATCH_SIZE': batch_size}

    return data, scaler
