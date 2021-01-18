import pandas as pd
import numpy as np
from dateutil.parser import parse
import matplotlib as mpl
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from datetime import datetime
import time
import random
from scipy.interpolate import interp1d
import statsmodels.api as sm
import seaborn as sns
import itertools
from statsmodels.tsa.stattools import adfuller
from numpy import mean
import math
import os
from statsmodels.graphics.api import qqplot
from statsmodels.stats.stattools import durbin_watson #DW检验

# 绘图plt设置中文和负号正常显示
plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

def get_data_dir(dataname):
    father_dir = os.path.abspath('..')
    data_dir = father_dir + '\data'+f'\{dataname}'
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

def plot_freq(df,KPI,num):
    df_freq = pd.DataFrame(index=df[KPI][0:num].index)
    df_freq['天数'] = [f'第{int(d / 24) + 1}天' for d in df[KPI][0:num].index]
    df_freq['小时'] = [f'{(d) % 24 + 1}' for d in df[KPI][0:2207].index]
    df_freq['values'] = df[KPI][0:num]

    # 绘图
    week_num = df_freq['天数'].unique()
    np.random.seed(100)
    mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(week_num), replace=False)
    plt.figure(figsize=(16, 12), dpi=80)
    for i, y in enumerate(week_num[0:-1]):
        plt.plot('小时', 'values', data=df_freq[df_freq["天数"] == y], color=mycolors[i], label=y)
    plt.gca().set(xlim=(0, 23), ylim=(70, 95), ylabel='$Memory Load$', xlabel='$Hour$')
    plt.yticks(fontsize=12, alpha=.7)
    plt.show()

def ADF_is_diff(adf):
    P_value = adf[0]
    p1 = adf[4]['1%']
    p5 = adf[4]['5%']
    p10 = adf[4]['10%']
    if P_value < p1 and P_value < p5 and P_value < p10:
        return False
    else:
        return True

def plot_BIC(timeSeries,p_min, p_max, q_min, q_max, d_min, d_max):
    results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                               columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])
    num = 1
    for p, d, q in itertools.product(range(p_min, p_max + 1),
                                     range(d_min, d_max + 1),
                                     range(q_min, q_max + 1)):
        if p == 0 and d == 0 and q == 0:
            results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
            continue

        try:
            model = sm.tsa.ARIMA(timeSeries, order=(p, d, q), #enforce_stationarity=False,enforce_invertibility=False,
                                 )
            results = model.fit()
            num += 1
            print(f"循环{num}次")
            results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
        except:
            continue
    results_bic = results_bic[results_bic.columns].astype(float)
    return results_bic

def evaluate(truth, predict, n=None, p=None):
    error = []
    squaredError = []   # 差值平方
    absError = []       # 绝对误差
    truthDeviation = []     # 真实值与均值的差平方
    percentError = []       # 误差百分比
    for i in range(len(truth)):
        error.append(truth[i]-predict[i])
        percentError.append(abs(truth[i]-predict[i])/truth[i])
    for val in error:
        squaredError.append(val**2)
        absError.append(abs(val))
    truthMean = sum(truth) / len(truth)
    for val in truth:
        truthDeviation.append((val - truthMean) ** 2)

    MSE = sum(squaredError)/len(squaredError)   # 均方误差MSE
    RMSE = math.sqrt(MSE)                       # 均方根误差RMSE
    MAE = sum(absError)/len(absError)           # 平均绝对误差MAE
    R2 = 1 - sum(squaredError)/sum(truthDeviation)  # 决定系数
    MAPE = sum(percentError)/len(percentError)  # 平均百分比误差MAPE

    R2_adj = 1 - ((1-R2)(n-1))/(n-p-1)  # 矫正决定系数
    return MSE,RMSE,MAE,MAPE,R2,R2_adj

if __name__ == "__main__":
    # 参数
    filename = "服务器性能数据.xlsx"
    KPI = "内存负载"
    window = 24   # K-近邻插值的窗口大小
    freq = 24   # 周期大小
    is_lack = True
    is_delete_freq = True
    forecast_index = 2046   # 预测时刻索引
    current_time = 6        # 预测时刻
    h = 12                  # 预测未来h步
    test_method = "DW"      # "DW"or"QQ"

    # 1.读取data数据
    df_original = read_file(filename)
    # 2.在数据中建立深拷贝副本
    df = df_original.copy(deep=True)
    # 3.观察原始序列图
    # plot_df(df, x=df.index, y=df[KPI], title=KPI)
    # 4.缺失值补全
    if is_lack:
        df[KPI] = knn_mean(df[KPI], window)
    # 5.观察周期性
    # num = len(df[KPI])
    # plot_freq(df, KPI, num)
    # 6.去除周期性因素
    seasonal_series = []
    if is_delete_freq:
        result_add = sm.tsa.seasonal_decompose(df[KPI], model='additive', period=24, extrapolate_trend='freq')
        for i in (result_add.seasonal)[0:freq].values:
            seasonal_series.append(i)
        df[KPI] -= result_add.seasonal
    # 7.划分预测区间
    train = df[KPI][0:forecast_index]
    test = df_original[KPI][forecast_index: h+forecast_index]
    # 8.观察时间序列平稳性,不满足,则进行差分
    train_temp = train.copy(deep=True)
    adftest = ts.adfuller(train_temp)
    if ADF_is_diff(adftest):
        train_temp = train_temp.diff(1)
        adftest = ts.adfuller(train_temp[1:])
        if ADF_is_diff(adftest):
            print("1次差分后也不满足平稳性要求")
        else:
            print("1次差分后时间序列平稳")
    else:
        print("时间序列平稳")
    # 8.差分后时间序列timeSeries的ACF和 PACF观察图
    KPI_dif = train_temp[1:]
    plot_acf(KPI_dif).show()
    plot_pacf(KPI_dif).show()

    # # 9.1 手动定阶
    # p_min, p_max = 3, 4
    # q_min, q_max = 3, 4
    # d_min, d_max = 0, 1
    # results_bic = plot_BIC(KPI_dif,p_min, p_max, q_min, q_max, d_min, d_max)
    # fig, ax = plt.subplots(figsize=(10, 8))
    # ax = sns.heatmap(results_bic,
    #                  mask=results_bic.isnull(),
    #                  ax=ax,
    #                  annot=True,
    #                  fmt='.2f',
    #                  )
    # ax.set_title('BIC')
    # plt.show()

    # 10. 建模
    model = sm.tsa.ARIMA(train.values, order=(7, 1, 4))
    result = model.fit()
    forecast, stderr, conf = result.forecast(12)
    resid = result.resid
    # 11.检验
    if test_method == "QQ":
        plt.figure(figsize=(12, 8))
        qqplot(resid, line='q', fit=True)
    if test_method == "DW":
        print('D-W检验值为{}'.format(durbin_watson(resid)))
    # 12.评价指标
    if is_delete_freq:
        MSE,RMSE,MAE,MAPE,R2,R2_adj = evaluate(test, forecast + seasonal_series[current_time: current_time+h],n=168, p=25)
    else:
        MSE, RMSE, MAE, MAPE, R2, R2_adj = evaluate(test, forecast,n=168, p=25)
    print(f"MSE:{MSE}\nRMSE:{RMSE}\nMAE:{MAE}\nMAPE:{MAPE}\nR2:{R2}\nR2_adj:{R2_adj}")
