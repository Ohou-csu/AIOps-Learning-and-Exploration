#!/usr/bin/python 3.8
# -*-coding:utf-8-*-

'''
Utility functions
'''
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import math
import os
from sklearn.metrics import mean_squared_error  # 均方误差
from sklearn.metrics import mean_absolute_error  # 平方绝对误差
from sklearn.metrics import r2_score  # R square

# 1.预处理函数
def get_data_dir(dataname):
    father_dir = os.path.abspath('..')
    data_dir = father_dir + f'/data/{dataname}'
    return data_dir

def read_file(filename):
    temp = filename.split(".")
    if temp[1] == "xlsx" or temp[1] == "xls":
        return pd.read_excel(get_data_dir(filename), index_col=0)
    elif temp[1] == "csv":
        return pd.read_csv(get_data_dir(filename), index_col=0)
    else:
        return "文件不存在或格式不符"

def plot_df( x, y, title="", xlabel='Index', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:blue')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

# k近邻法
def knn_mean(ts, n):
    out = np.copy(ts)
    for i, val in enumerate(ts):
        if np.isnan(val):
            n_by_2 = np.ceil(n / 2)
            lower = np.max([0, int(i - n_by_2)])
            upper = np.min([len(ts) + 1, int(i + n_by_2)])
            ts_near = np.concatenate([ts[lower:i], ts[i:upper]])
            out[i] = np.nanmean(ts_near)
    return out

# 建立自回归预测矩阵
def Autoregressive_matrix(data,num_hour, pred_h):
    dataframe = pd.DataFrame()
    for i in range(num_hour-1,0,-1):
        dataframe['t-'+str(i)] = data.shift(i)
    dataframe['t'] = data.values
    for i in range(1,pred_h+1):
        dataframe['t+'+str(i)] = data.shift(periods=-i, axis=0)
    all_data = dataframe[num_hour:-pred_h]
    # print(all_data)
    return all_data

def dataloader(dataframe, Dataset_Ratio, is_Shuffle,pred_h):
    if is_Shuffle:
        all_data = shuffle(dataframe)
    else:
        all_data = dataframe
    a, b, c = Dataset_Ratio
    ratio1 = a/(a+b+c)
    ratio2 = (a+b)/(a+b+c)
    var1 = int(len(all_data) * ratio1)
    var2 = int(len(all_data) * ratio2)
    train_data = all_data[0:var1]
    train_truth = all_data.iloc[0:var1, -pred_h:]
    validate_data = all_data[var1:var2]
    validate_truth = all_data.iloc[var1:var2, -pred_h:]
    test_data = all_data[var2:]
    test_truth = all_data.iloc[var2:, -pred_h:]
    return train_data,train_truth,validate_data,validate_truth,test_data,test_truth

def NormalizeAndToTensor(data, num_hour, pred_h, device):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data.values.reshape(-1, 1))
    data_normalized = data_normalized.reshape(-1, num_hour + pred_h)
    data_X = torch.Tensor(data_normalized[:, 0:-pred_h].reshape(-1, int(num_hour / 24), 24)).to(device)
    data_Y = torch.Tensor(data_normalized[:, -pred_h:].reshape(-1, 1, pred_h)).to(device)
    return data_X, data_Y

# 3.验证函数
def mape(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值
    返回:
    mape -- MAPE 评价指标
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape

# 保留几位小数
def get_two_float(f_str, n):
    f_str = str(f_str)      # f_str = '{}'.format(f_str) 也可以转换为字符串
    a, b, c = f_str.partition('.')
    c = (c+"0"*n)[:n]       # 如论传入的函数有几位小数，在字符串后面都添加n为小数0
    return ".".join([a, c])
