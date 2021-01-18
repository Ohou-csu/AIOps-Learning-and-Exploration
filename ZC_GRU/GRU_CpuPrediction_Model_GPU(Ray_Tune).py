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
from ray import tune
from itertools import product
import json

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer):
        super(GRU, self).__init__()
        self.layer1 = nn.GRU(input_size, hidden_size, num_layer, batch_first=True)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        x = x[:, [-1], :]
        b, s, h = x.size()
        x = x.view(-1, h)
        x = self.layer2(x)
        x = x.view(b, s, -1)
        return x

def train(train_X, train_Y, validate_X, validate_Y, model, criterion, optimizer, epoch, early_stopping, lr_down,device):
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
    modelName = "GRU"
    num_hour = config["num_hour"]  # 历史数据个数144
    pred_h = config["pred_h"]  # 预测步数12

    # GRU参数
    input_size = 24
    criterion = nn.MSELoss()
    epoch = config["epoch"]                                #1000
    patience = config["patience"]                      #15
    lr = config["lr"]                #1e-2
    hidden_size = config["hidden_size"]           #64

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
        (dataframe, (6, 2, 2), is_Shuffle=True, pred_h=pred_h)
    max_value = max(train_data.max().values)
    min_value = min(train_data.min().values)
    # range_value = max_value - min_value
    # 6.归一化并转Tensor
    train_X, train_Y = zc_util.NormalizeAndToTensor(train_data, num_hour, pred_h, device)
    validate_X, validate_Y = zc_util.NormalizeAndToTensor(validate_data, num_hour, pred_h, device)
    test_X, test_Y = zc_util.NormalizeAndToTensor(test_data, num_hour, pred_h, device)
    # 7.建立GRU模型以及模型参数
    model = GRU(input_size, hidden_size=hidden_size, output_size=pred_h, num_layer=2).to(device)
    # print(model_lstm)
    optimizer_lstm = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience, verbose=True)
    # 8.开始训练
    # train_X, train_Y, validate_X, validate_Y, model, criterion, optimizer, epoch, early_stopping, lr_down
    epochs = []
    losses = []
    for i in range(10):
        epochs, losses = train(train_X=train_X, train_Y=train_Y, validate_X=validate_X,
                                     validate_Y=validate_Y, model=model, criterion=criterion,
                                     optimizer=optimizer_lstm, epoch=epoch, early_stopping=early_stopping,
                                     lr_down=None,device=device)
        # 9.结束并保存
        # zc_util.plot_df(epochs, losses_LSTM)
        # print('Finished Training')

        torch.save(model, f'{modelName}.pkl')
        model2 = torch.load(f'{modelName}.pkl')

        # 10.验证
        outputs = model2(test_X)
        predict = (outputs * (max_value - min_value) + min_value).squeeze().detach().cpu().numpy()  # (372*1*12)
        truth = test_truth.values.reshape(-1, pred_h)  # 374*1
        MSE_test = mean_squared_error(truth, predict)
        MAE_test = mean_absolute_error(truth, predict)
        # print(f"测试集整体MSE: {MSE_test}")
        # print(f"测试集整体RMSE: {np.sqrt(MSE_test)}")
        # print(f"测试集整体MAE: {MAE_test}")
        # print(f"测试集整体MAPE: {zc_util.mape(truth, predict)}")

        Verification = {
            "MSE_test": zc_util.get_two_float(MSE_test, 6),
            "RMSE_test": zc_util.get_two_float(np.sqrt(MSE_test), 6),
            "MAE_test": zc_util.get_two_float(MAE_test, 6),
            "MAPE_test": zc_util.get_two_float(zc_util.mape(truth, predict), 6)
        }

        outputs = model(train_X)
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
        config["input_size"] = input_size
        data = {
            "modelName": modelName,
            "Parameter": config,
            "Raw_Data": KPI,
            "epochs": epochs[-1],
            "losses": losses,
            "Verification": Verification,
            "training": training
        }
        return data

if __name__ == '__main__':
    start = time.time()
    modelName = "GRU"
    search_space = {
        "num_hour": [72,168,336],
        "pred_h": [1,6,12],
        "epoch": [5000],
        "input_size": [12, 24],
        "hidden_size": [64, 128, 256],
        "patience": [20,25,30],
        "lr": [1e-2, 1e-3],
    }

    analysis = tune.run(search, config=search_space)




    # arrs = list(product(*search_space.values()))
    # # print(arrs)
    # datas = []
    # print(f"总搜索次数为:{len(arrs)}")
    # for index, arr in enumerate(arrs):
    #     search_config = dict(zip(list(search_space.keys()), list(arr)))
    #     search_data = search(search_config, modelName)
    #     print(f"已搜索{index+1}次")
    #     datas.append(search_data)
    # datas_json = json.dumps(datas)
    # with open(f"data_{modelName}.txt", "w+", encoding="UTF-8") as f:
    #     f.write(str(datas_json))
    #     f.close()
    end = time.time()
    time = end - start
    print(f'运行时长为:{int(time)}s')


