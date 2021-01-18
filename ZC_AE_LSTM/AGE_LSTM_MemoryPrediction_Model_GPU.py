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
from pytorchtools import EarlyStopping
import math
import os
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square
import zc_util
import ray
from itertools import product

class AE(nn.Module):
    def __init__(self,input_size,hidden_size,hat_size,num_hour):
        super(AE, self).__init__()
        self.GRU_layer1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output_linear1 = nn.Linear(in_features=hidden_size, out_features=hat_size)

        self.GRU_layer2 = nn.GRU(input_size=int(hat_size/(num_hour/input_size)), hidden_size=hidden_size, batch_first=True)
        self.output_linear2 = nn.Linear(in_features=hidden_size, out_features=num_hour)
        self.hidden = None

    def forward(self, x):
        x, self.hidden = self.GRU_layer1(x)
        b1, s1, h1 = x.size()
        x = x[:, [s1-1], :]
        x = x.view(-1,h1)
        x = self.output_linear1(x)
        x_hat = x.view(b1, s1, -1)

        x, self.hidden = self.GRU_layer2(x_hat)
        b2, s2, h2 = x.size()
        x = x[:, [s2 - 1], :]
        x = x.view(-1, h2)
        x = self.output_linear2(x)
        x = x.view(b2, s2, -1)
        return x,x_hat

class lstm(nn.Module):
    def __init__(self, input_size=16, hidden_size=100, output_size=1, num_layer=2):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        x = x[:, [-1], :]
        b, s, h = x.size()
        x = x.view(-1, h)
        x = self.layer2(x)
        x = x.view(b, s, -1)
        return x

def train_AE(train_X, train_Y, model, criterion, optimizer, epoch, lr_down,device):
    ep_AE = []
    losses_AE = []
    lr_list_AE = []
    for i in range(1, epoch + 1):
        x = Variable(train_X).to(device)
        # 前向传播
        out, out_hat = model(x)
        loss = criterion(out, x)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if i % 10 == 0:  # 每 10 次输出结果
        #     print('Epoch: {}, Loss: {:.8f}'.format(i, loss.item()))
        ep_AE.append(i)
        losses_AE.append(zc_util.get_two_float(loss.item(), 6))
    return ep_AE, losses_AE

def train_LSTM(train_X, train_Y, validate_X, validate_Y, model, criterion, optimizer, epoch, early_stopping, lr_down,device):
    # 14.开始训练
    ep_Ls = []
    losses_Ls = []
    lr_list_LS = []
    for e in range(1, epoch + 1):
        var_x = Variable(train_X).to(device)
        var_y = Variable(train_Y).to(device)
        # 前向传播
        out = model(var_x)
        loss = criterion(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        valid_output = model(validate_X)
        valid_loss = criterion(valid_output, validate_Y)

        # if e % 10 == 0:  # 每 10 次输出结果
        #     print('Epoch: {}, Loss: {:.8f}, VA_Loss: {:.8f}'.format(e, loss.item(), valid_loss.item()))
        ep_Ls.append(e)
        losses_Ls.append(zc_util.get_two_float(loss.item(), 6))
        early_stopping(valid_loss, model)

        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            # print("Early stopping")
            # 结束模型训练
            break
    return ep_Ls, losses_Ls
        # if lr_down
        # if (e+1)%120 == 0:
        #     for p in optimizer.param_groups:
        #         p['lr'] *= 0.1
        # lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

def search(config):
    # 参数设置
    filename = "服务器性能数据.xlsx"
    KPI = "主机CPU平均负载"
    num_hour = config["num_hour"]  # 历史数据个数144
    pred_h = config["pred_h"]  # 预测步数12

    # GAE参数
    criterion_AE = nn.MSELoss()
    epoch_AE = config["epoch_AE"]         # 3000
    lr_AE = config["lr_AE"]                 # 1e-2
    input_size_AE = config["input_size_AE"]             #24
    hiddenSize_AE = config["hiddenSize_AE"]            #256
    hat_size_AE = config["hat_size_AE"]                  # 84

    # LSTM参数
    input_size_lstm = int(hat_size_AE / (num_hour / input_size_AE))
    criterion_lstm = nn.MSELoss()
    epoch_lstm = config["epoch_lstm"]                                #1000
    patience = config["patience"]                      #15
    lr_lstm = config["lr_lstm"]                #1e-2
    LSTM_hidden_size = config["LSTM_hidden_size"]           #64

    # 归一化到0~1
    device = torch.device("cuda")
    np.random.seed(113)


    # 1.读取data数据
    df_original = zc_util.read_file(filename)
    # 2.在数据中建立深拷贝副本
    df = df_original.copy(deep=True)
    # 3.使用k-近邻法填补缺失值
    df[KPI] = zc_util.knn_mean(df[KPI], 24)
    # 4.建立自回归预测矩阵
    dataframe = zc_util.Autoregressive_matrix(df[KPI], num_hour=num_hour, pred_h=pred_h)
    # 5.划分测试集和训练集
    train_data, train_truth, validate_data, validate_truth, test_data, test_truth = zc_util.dataloader \
        (dataframe, (6, 2, 2), is_Shuffle=False, pred_h=pred_h)
    max_value = max(train_data.max().values)
    min_value = min(train_data.min().values)
    # range_value = max_value - min_value
    # 6.归一化并转Tensor
    train_X, train_Y = zc_util.NormalizeAndToTensor(train_data, num_hour, pred_h, device)
    validate_X, validate_Y = zc_util.NormalizeAndToTensor(validate_data, num_hour, pred_h, device)
    test_X, test_Y = zc_util.NormalizeAndToTensor(test_data, num_hour, pred_h, device)
    # 7.建模以及模型参数
    model_AE = AE(24, hidden_size=hiddenSize_AE, hat_size=hat_size_AE, num_hour=num_hour).to(device)
    # print(model_AE)
    optimizer = torch.optim.Adam(model_AE.parameters(), lr=lr_AE)
    # 8.开始训练
    epochs_AE, losses_AE = train_AE(train_X, train_Y, model=model_AE,
                                 criterion=criterion_AE, optimizer=optimizer, epoch=epoch_AE, lr_down=None,device=device)

    # 10.保存AE模型
    torch.save(model_AE, 'AE.pkl')
    # 11.加载AE模型
    model_AE2 = torch.load('AE.pkl')

    # 12.生成隐变量
    outputs, outputs_hat = model_AE2(train_X)
    train_X2 = outputs_hat.to(device)
    outputs, outputs_hat = model_AE2(validate_X)
    validate_X2 = outputs_hat.to(device)
    outputs, outputs_hat = model_AE2(test_X)
    test_X2 = outputs_hat.to(device)

    # 13.建立LSTM模型
    model_lstm = lstm(input_size_lstm, hidden_size=LSTM_hidden_size, output_size=pred_h, num_layer=2).to(device)
    # print(model_lstm)
    optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=lr_lstm)
    early_stopping = EarlyStopping(patience, verbose=True)
    # 8.开始训练
    # train_X, train_Y, validate_X, validate_Y, model, criterion, optimizer, epoch, early_stopping, lr_down
    epochs_LSTM, losses_LSTM = train_LSTM(train_X=train_X2, train_Y=train_Y, validate_X=validate_X2,
                                     validate_Y=validate_Y, model=model_lstm, criterion=criterion_lstm,
                                     optimizer=optimizer_lstm, epoch=epoch_lstm, early_stopping=early_stopping,
                                     lr_down=None,device=device)
    # 15.结束并保存
    # zc_util.plot_df(epochs, losses_LSTM)
    # print('Finished Training')

    torch.save(model_lstm, 'LSTM.pkl')
    model_lstm2 = torch.load('LSTM.pkl')

    # 16.验证
    outputs = model_lstm2(test_X2)
    predict = (outputs * (max_value - min_value) + min_value).squeeze().detach().cpu().numpy()  # (372*1*12)
    truth = test_truth.values.reshape(-1, pred_h)  # 374*1
    MSE_test = mean_squared_error(truth, predict)
    MAE_test = mean_absolute_error(truth, predict)
    # print(f"测试集整体MSE: {MSE_test}")
    # print(f"测试集整体RMSE: {np.sqrt(MSE_test)}")
    # print(f"测试集整体MAE: {MAE_test}")
    # print(f"测试集整体MAPE: {zc_util.mape(truth, predict)}")
    #
    # print("#########################")
    Verification = {
        "MSE_test": zc_util.get_two_float(MSE_test, 6),
        "RMSE_test": zc_util.get_two_float(np.sqrt(MSE_test), 6),
        "MAE_test": zc_util.get_two_float(MAE_test, 6),
        "MAPE_test": zc_util.get_two_float(zc_util.mape(truth, predict), 6)
    }

    outputs = model_lstm2(train_X2)
    predict2 = (outputs * (max_value - min_value) + min_value).squeeze().detach().cpu().numpy()
    truth2 = train_truth.values.reshape(-1, pred_h)
    MSE2_test = mean_squared_error(truth2, predict2)
    MAE2_test = mean_absolute_error(truth2, predict2)
    # print(f"测试集整体MSE: {MSE2_test}")
    # print(f"训练集整体RMSE: {np.sqrt(MSE2_test)}")
    # print(f"训练集整体MAE: {MAE2_test}")
    # print(f"训练集整体MAPE: {zc_util.mape(truth2, predict2)}")
    training = {
        "MSE_train": zc_util.get_two_float(MSE2_test, 6),
        "RMSE_train": zc_util.get_two_float(np.sqrt(MSE2_test), 6),
        "MAE_train": zc_util.get_two_float(MAE2_test, 6),
        "MAPE_train": zc_util.get_two_float(zc_util.mape(truth2, predict2), 6)
    }


    # 17.保存验证
    config["input_size_lstm"] = input_size_lstm
    data = {
        "Parameter": config,
        "Raw_Data": KPI,
        "epochs_AE": epochs_AE[-1],
        "losses_AE": losses_AE,
        "epochs_LSTM": epochs_LSTM[-1],
        "losses_LSTM": losses_LSTM,
        "Verification": Verification,
        "training": training
    }
    return data

if __name__ == '__main__':
    start = time.time()
    search_space = {
        "num_hour": [36, 72, 168, 336],
        "pred_h": [1, 6, 12, 24],
        "epoch_AE": [3000],
        "lr_AE": [1e-2],
        "input_size_AE": [24],
        "hiddenSize_AE": [256],
        "hat_size_AE": [84, 42],
        "epoch_lstm": [5000],
        "patience": [40],
        "lr_lstm": [1e-2, 1e-3],
        "LSTM_hidden_size": [64]
    }
    arrs = list(product(*search_space.values()))
    datas = []
    print(f"总搜索次数为:{len(arrs)}")
    for index, arr in enumerate(arrs):
        search_config = dict(zip(list(search_space.keys()), list(arr)))
        search_data = search(search_config)
        print(f"已搜索{index+1}次")
        datas.append(search_data)
    with open("data_2_without_shuffle2.txt", "w+", encoding="UTF-8") as f:
        f.write(str(datas))
        f.close()
    end = time.time()
    time = end - start
    print(f'运行时长为:{int(time)}s')


