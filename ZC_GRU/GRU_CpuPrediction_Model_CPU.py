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

start = time.time()

# 归一化到0~1
scaler = MinMaxScaler(feature_range=(0, 1))

def get_data_dir(dataname):
    father_dir = os.path.abspath('..')
    data_dir = father_dir + '/data'+f'/{dataname}'
    return data_dir

def read_file(filename):
    temp = filename.split(".")
    if temp[1] == "xlsx" or temp[1] == "xls":
        return pd.read_excel(get_data_dir(filename), index_col=0)
    elif temp[1] == "csv":
        return pd.read_csv(get_data_dir(filename), index_col=0)
    else:
        return "文件不存在或格式不符"

def plot_df(df, x, y, title="", xlabel='Index', ylabel='Value', dpi=100):
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

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer):
        super(GRU, self).__init__()
        self.layer1 = nn.GRU(input_size, hidden_size, num_layer, batch_first=True)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #         x,_ = self.layer1(x)
        #         b, s, h = x.size()
        #         x = x.reshape(-1,s*h)
        #         x = self.layer2(x)
        #         x = x.reshape(b, 1, 1)
        x, _ = self.layer1(x)
        x = x[:, [13], :]
        b, s, h = x.size()
        x = x.view(-1, h)
        x = self.layer2(x)
        x = x.view(b, s, -1)
        return x


# 参数设置
filename = "服务器性能数据.xlsx"
KPI = "CPU平均负载"
num_hour = 336      # 历史数据个数
pred_h = 1     # 预测步数

# 1.读取data数据
df_original = read_file(filename)
# 2.在数据中建立深拷贝副本
df = df_original.copy(deep=True)

# 3.使用k-近邻法填补缺失值
df["主机CPU平均负载"] = knn_mean(df["主机CPU平均负载"], 24)

# 4.建立自回归预测矩阵
data = df["主机CPU平均负载"]
dataframe = pd.DataFrame()
for i in range(num_hour-1,0,-1):
    dataframe['t-'+str(i)] = data.shift(i)
dataframe['t'] = data.values
for i in range(1,pred_h+1):
    dataframe['t+'+str(i)] = data.shift(periods=-i, axis=0)
# print(dataframe)
# 5.划分测试集和训练集
np.random.seed(113)
all_data = dataframe[num_hour:-pred_h]
all_data = shuffle(all_data)

var1 = int(len(all_data)*0.6)
var2 = int(len(all_data)*0.8)
train_data = all_data[0:var1]
train_truth = all_data.iloc[0:var1,-pred_h:]
validate_data = all_data[var1:var2]
validate_truth = all_data.iloc[var1:var2,-pred_h:]
test_data = all_data[var2:]
test_truth = all_data.iloc[var2:,-pred_h:]

max_value = max(train_data.max().values)
min_value = min(train_data.min().values)
ch = max_value - min_value

# 6.归一化
train_data_normalized = scaler.fit_transform(train_data.values.reshape(-1, 1))
train_data_normalized = train_data_normalized.reshape(-1,num_hour+pred_h)

validate_data_normalized = scaler.transform(validate_data.values.reshape(-1, 1))
validate_data_normalized = validate_data_normalized.reshape(-1,num_hour+pred_h)

test_data_normalized = scaler.transform(test_data.values.reshape(-1, 1))
test_data_normalized = test_data_normalized.reshape(-1,num_hour+pred_h)

# 7.转Tensor
train_X = torch.Tensor(train_data_normalized[:,0:-pred_h].reshape(-1, int(num_hour/24), 24))
train_Y = torch.Tensor(train_data_normalized[:,-pred_h:].reshape(-1, 1, pred_h))

validate_X = torch.Tensor(validate_data_normalized[:,0:-pred_h].reshape(-1, int(num_hour/24), 24))
validate_Y = torch.Tensor(validate_data_normalized[:,-pred_h:].reshape(-1, 1, pred_h))

test_X = torch.Tensor(test_data_normalized[:,0:-pred_h].reshape(-1, int(num_hour/24), 24))
test_Y = torch.Tensor(test_data_normalized[:,-pred_h:].reshape(-1, 1, pred_h))

# 8.建模以及模型参数
model = GRU(24, 128, pred_h, 2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
epoch_n = 1000
# 初始化 early_stopping 对象
# 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
patience = 20
early_stopping = EarlyStopping(patience, verbose=True)

# 9.开始训练
ep = []
losses = []
lr_list = []
for e in range(1, epoch_n + 1):
    var_x = Variable(train_X)
    var_y = Variable(train_Y)
    # 前向传播
    out = model(var_x)
    loss = criterion(out, var_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    valid_output = model(validate_X)
    valid_loss = criterion(valid_output, validate_Y)

    if e % 10 == 0:  # 每 10 次输出结果
        print('Epoch: {}, Loss: {:.8f}, VA_Loss: {:.8f}'.format(e, loss.item(), valid_loss.item()))
    ep.append(e)
    losses.append(loss.item())

    early_stopping(valid_loss, model)

    # 若满足 early stopping 要求
    if early_stopping.early_stop:
        print("Early stopping")
        # 结束模型训练
        break
print('Finished Training')
end = time.time()
time = end - start
print(f'运行时长为:{int(time)}s')

# 10.绘制loss变化图
plot_df(df, x=ep, y= losses, title='LOSS')

# 11.保存模型
torch.save(model, 'net.pkl')
# 12.加载模型
model2 = torch.load('net.pkl')

# 13.验证
outputs = model2(test_X)
predict = (outputs*(max_value-min_value) + min_value).squeeze().detach().numpy()     # (372*1*12)
truth = test_truth.values.reshape(-1, pred_h)
MSE_test = mean_squared_error(truth, predict)
MAE_test = mean_absolute_error(truth, predict)
print(f"测试集整体MSE: {MSE_test}")
print(f"测试集整体RMSE: {np.sqrt(MSE_test)}")
print(f"测试集整体MAE: {MAE_test}")
print(f"测试集整体MAPE: {mape(truth, predict)}")
print("#########################")


outputs = model2(train_X)
loss = criterion(outputs, train_Y)
predict2 = (outputs*(max_value-min_value) + min_value).squeeze().detach().numpy()
truth2 = train_truth.values.reshape(-1, pred_h)
MSE2_test= mean_squared_error(truth2, predict2)
MAE2_test = mean_absolute_error(truth2, predict2)
print(f"测试集整体MSE: {MSE2_test}")
print(f"训练集整体RMSE: {np.sqrt(MSE2_test)}")
print(f"训练集整体MAE: {MAE2_test}")
print(f"训练集整体MAPE: {mape(truth2, predict2)}")

# 单步
# 运行时长为:208s
# 测试集整体MSE: 0.17708381468441506
# 测试集整体RMSE: 0.420813277695007
# 测试集整体MAE: 0.3201493875775493
# 测试集整体MAPE: 10.425444106991275
# #########################
# 测试集整体MSE: 0.18957402351570943
# 训练集整体RMSE: 0.43540099163381496
# 训练集整体MAE: 0.32413083328595027
# 训练集整体MAPE: 10.508447091235565

#  单步
# 运行时长为:233s
# 测试集整体MSE: 0.16905632428814987
# 测试集整体RMSE: 0.4111645951296754
# 测试集整体MAE: 0.3003216803129529
# 测试集整体MAPE: 9.277163117884966
# #########################
# 测试集整体MSE: 0.1940520373677871
# 训练集整体RMSE: 0.4405133793289224
# 训练集整体MAE: 0.3142916408278913
# 训练集整体MAPE: 9.62223000042344

# 12步
# 运行时长为:281s
# 测试集整体MSE: 0.2680540010941607
# 测试集整体RMSE: 0.5177393177016409
# 测试集整体MAE: 0.37310739345536587
# 测试集整体MAPE: 11.510203741591212
# #########################
# 测试集整体MSE: 0.15849217513028172
# 训练集整体RMSE: 0.398110757365688
# 训练集整体MAE: 0.29647410060151796
# 训练集整体MAPE: 9.48747768024728

# 12步
# 运行时长为:314s
# 测试集整体MSE: 0.2621011439447375
# 测试集整体RMSE: 0.5119581466728872
# 测试集整体MAE: 0.3721397018894495
# 测试集整体MAPE: 11.275024044347507
# #########################
# 测试集整体MSE: 0.16921210652951987
# 训练集整体RMSE: 0.4113539917510463
# 训练集整体MAE: 0.3028333407418222
# 训练集整体MAPE: 9.406582216547264



