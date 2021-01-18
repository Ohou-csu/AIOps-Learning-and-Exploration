import torch
import zc_util
import util
import pandas as pd
import os
import numpy as np
from torch import nn
import argparse
from torch.optim import Adam
import random
from progressbar import *
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import os

class Gaussian(nn.Module):
    def __init__(self, hidden_size, output_size):
        '''
        Gaussian Likelihood Supports Continuous Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(Gaussian, self).__init__()
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)

        # initialize weights
        # nn.init.xavier_uniform_(self.mu_layer.weight)
        # nn.init.xavier_uniform_(self.sigma_layer.weight)

    def forward(self, h):
        _, hidden_size = h.size()
        sigma_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        sigma_t = sigma_t.squeeze(0)
        mu_t = self.mu_layer(h).squeeze(0)
        return mu_t, sigma_t

class NegativeBinomial(nn.Module):

    def __init__(self, input_size, output_size):
        '''
        Negative Binomial Supports Positive Count Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(NegativeBinomial, self).__init__()
        self.mu_layer = nn.Linear(input_size, output_size)
        self.sigma_layer = nn.Linear(input_size, output_size)

    def forward(self, h):
        _, hidden_size = h.size()
        alpha_t = torch.log(1 + torch.exp(self.sigma_layer(h))) + 1e-6
        mu_t = torch.log(1 + torch.exp(self.mu_layer(h)))
        return mu_t, alpha_t

def gaussian_sample(mu, sigma):
    '''
    Gaussian Sample
    Args:
    ytrue (array like)
    mu (array like)
    sigma (array like): standard deviation

    gaussian maximum likelihood using log
        l_{G} (z|mu, sigma) = (2 * pi * sigma^2)^(-0.5) * exp(- (z - mu)^2 / (2 * sigma^2))
    '''
    # likelihood = (2 * np.pi * sigma ** 2) ** (-0.5) * \
    #         torch.exp((- (ytrue - mu) ** 2) / (2 * sigma ** 2))
    # return likelihood
    gaussian = torch.distributions.normal.Normal(mu, sigma)
    ypred = gaussian.sample(mu.size())
    return ypred

def negative_binomial_sample(mu, alpha):
    '''
    Negative Binomial Sample
    Args:
    ytrue (array like)
    mu (array like)
    alpha (array like)

    maximuze log l_{nb} = log Gamma(z + 1/alpha) - log Gamma(z + 1) - log Gamma(1 / alpha)
                - 1 / alpha * log (1 + alpha * mu) + z * log (alpha * mu / (1 + alpha * mu))

    minimize loss = - log l_{nb}

    Note: torch.lgamma: log Gamma function
    '''
    var = mu + mu * mu * alpha
    ypred = mu + torch.randn(mu.size()) * torch.sqrt(var)
    return ypred

def batch_generator(X, y, num_obs_to_train, seq_len, batch_size):
    '''
    Args:
    X (array like): shape (num_samples, num_features, num_periods)
    y (array like): shape (num_samples, num_periods)
    num_obs_to_train (int):
    seq_len (int): sequence/encoder/decoder length
    batch_size (int)
    '''
    num_ts, num_periods, _ = X.shape
    if num_ts < batch_size:
        batch_size = num_ts
    t = random.choice(range(num_obs_to_train, num_periods-seq_len))
    batch = random.sample(range(num_ts), batch_size)
    X_train_batch = X[batch, t-num_obs_to_train:t, :]
    y_train_batch = y[batch, t-num_obs_to_train:t]
    Xf = X[batch, t:t+seq_len]
    yf = y[batch, t:t+seq_len]
    return X_train_batch, y_train_batch, Xf, yf

class DeepAR(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, lr=1e-3, likelihood="g"):
        super(DeepAR, self).__init__()

        # network
        self.input_embed = nn.Linear(1, embedding_size)
        self.encoder = nn.LSTM(embedding_size + input_size, hidden_size, \
                               num_layers, bias=True, batch_first=True)
        if likelihood == "g":
            self.likelihood_layer = Gaussian(hidden_size, 1)
        elif likelihood == "nb":
            self.likelihood_layer = NegativeBinomial(hidden_size, 1)
        self.likelihood = likelihood

    def forward(self, X, y, Xf):
        '''
        Args:
        X (array like): shape (num_time_series, seq_len, input_size)
        y (array like): shape (num_time_series, seq_len)
        Xf (array like): shape (num_time_series, horizon, input_size)
        Return:
        mu (array like): shape (batch_size, seq_len)
        sigma (array like): shape (batch_size, seq_len)
        '''
        # 如果X是numpy格式，转化为Tensor
        if isinstance(X, type(np.empty(2))):
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
            Xf = torch.from_numpy(Xf).float()
        num_ts, seq_len, _ = X.size()
        _, output_horizon, num_features = Xf.size()
        ynext = None
        ypred = []
        mus = []
        sigmas = []
        h, c = None, None
        for s in range(seq_len + output_horizon):
            if s < seq_len:
                ynext = y[:, s].view(-1, 1)
                yembed = self.input_embed(ynext).view(num_ts, -1)
                x = X[:, s, :].view(num_ts, -1)
            else:
                yembed = self.input_embed(ynext).view(num_ts, -1)
                x = Xf[:, s - seq_len, :].view(num_ts, -1)
            x = torch.cat([x, yembed], dim=1)  # num_ts, num_features + embedding
            inp = x.unsqueeze(1)
            if h is None and c is None:
                out, (h, c) = self.encoder(inp)  # h size (num_layers, num_ts, hidden_size)
            else:
                out, (h, c) = self.encoder(inp, (h, c))
            hs = h[-1, :, :]
            hs = F.relu(hs)
            mu, sigma = self.likelihood_layer(hs)
            mus.append(mu.view(-1, 1))
            sigmas.append(sigma.view(-1, 1))
            if self.likelihood == "g":
                ynext = gaussian_sample(mu, sigma)
            elif self.likelihood == "nb":
                alpha_t = sigma
                mu_t = mu
                ynext = negative_binomial_sample(mu_t, alpha_t)
            # if without true value, use prediction
            if s >= seq_len - 1 and s < output_horizon + seq_len - 1:
                ypred.append(ynext)
        ypred = torch.cat(ypred, dim=1).view(num_ts, -1)
        mu = torch.cat(mus, dim=1).view(num_ts, -1)
        sigma = torch.cat(sigmas, dim=1).view(num_ts, -1)
        return ypred, mu, sigma

if __name__ == '__main__':
    filename = "服务器性能数据.xlsx"

    # 1.读取data数据
    df_original = pd.read_excel(zc_util.get_data_dir(filename), parse_dates=["日期"], index_col=0)
    # 2.在数据中建立深拷贝副本
    data = df_original.copy(deep=True)

    data["主机CPU平均负载"] = zc_util.knn_mean(data["主机CPU平均负载"], 24)

    data["year"] = data["日期"].apply(lambda x: x.year)
    # dayofweek 查看一周的第几天
    data["day_of_week"] = data["日期"].apply(lambda x: x.dayofweek)
    data = data.loc[(data["日期"] >= '2020-04-01') & (data["日期"] <= '2020-05-01')]
    data["hour"] = data["日期"].apply(lambda x: int(str(x)[11:13]))
    print(data)
    features = ["hour", "day_of_week"]
    hours = data["hour"]
    dows = data["day_of_week"]
    # np.asarray 在参数对象是普通迭代序列时，asarray和array没有区别,都是将数组转换为ndarray对象
    # np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
    # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。
    X = np.c_[np.asarray(hours), np.asarray(dows)]
    num_features = X.shape[1]  # 2
    num_periods = len(data)  # 721
    X = np.asarray(X).reshape((-1, num_periods, num_features))
    y = np.asarray(data["主机CPU平均负载"]).reshape((-1, num_periods))

    # (num_ts=1, num_periods=744, num_features=2)
    num_ts, num_periods, num_features = X.shape
    model = DeepAR(num_features, embedding_size=10,hidden_size=50, num_layers=1, likelihood="nb")
    optimizer = Adam(model.parameters(), lr=1e-3)
    random.seed(2)
    Xtr, ytr, Xte, yte = util.train_test_split(X, y)

    losses = []
    cnt = 0
    yscaler = None
    # 数据放缩预处理
    # if args.standard_scaler:
    #     yscaler = util.StandardScaler()
    # elif args.log_scaler:
    #     yscaler = util.LogScaler()
    # elif args.mean_scaler:
    yscaler = util.MeanScaler()
    if yscaler is not None:
        ytr = yscaler.fit_transform(ytr)

    seq_len = 12
    num_obs_to_train = 168
    progress = ProgressBar()
    num_epoches = 100
    step_per_epoch = 3
    batch_size = 64
    likelihood = "nb"
    sample_size = 100  # 蒙特卡洛模拟次数
    show_plot = True

    for epoch in progress(range(num_epoches)):
        # print("Epoch {} starts...".format(epoch))
        for step in range(step_per_epoch):
            Xtrain, ytrain, Xf, yf = batch_generator(Xtr, ytr, num_obs_to_train, seq_len, batch_size)
            Xtrain_tensor = torch.from_numpy(Xtrain).float()
            ytrain_tensor = torch.from_numpy(ytrain).float()
            Xf = torch.from_numpy(Xf).float()
            yf = torch.from_numpy(yf).float()
            ypred, mu, sigma = model(Xtrain_tensor, ytrain_tensor, Xf)
            # ypred_rho = ypred
            # e = ypred_rho - yf
            # loss = torch.max(rho * e, (rho - 1) * e).mean()
            ## gaussian loss
            ytrain_tensor = torch.cat([ytrain_tensor, yf], dim=1)
            if likelihood == "g":
                loss = util.gaussian_likelihood_loss(ytrain_tensor, mu, sigma)
            elif likelihood == "nb":
                loss = util.negative_binomial_loss(ytrain_tensor, mu, sigma)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1

        # test
        mape_list = []
        # select skus with most top K
        X_test = Xte[:, -seq_len - num_obs_to_train:-seq_len, :].reshape((num_ts, -1, num_features))
        Xf_test = Xte[:, -seq_len:, :].reshape((num_ts, -1, num_features))
        y_test = yte[:, -seq_len - num_obs_to_train:-seq_len].reshape((num_ts, -1))
        yf_test = yte[:, -seq_len:].reshape((num_ts, -1))
        if yscaler is not None:
            y_test = yscaler.transform(y_test)
        result = []
        n_samples = sample_size
        for _ in tqdm(range(n_samples)):
            y_pred, _, _ = model(X_test, y_test, Xf_test)
            y_pred = y_pred.data.numpy()
            if yscaler is not None:
                y_pred = yscaler.inverse_transform(y_pred)
            result.append(y_pred.reshape((-1, 1)))

        #     result (100,12,1)
        result = np.concatenate(result, axis=1)  # (12,100)
        p50 = np.quantile(result, 0.5, axis=1)  # (12,)
        p90 = np.quantile(result, 0.9, axis=1)
        p10 = np.quantile(result, 0.1, axis=1)
        # yf_test(1,12)
        mape = util.MAPE(yf_test, p50)
        print("P50 MAPE: {}".format(mape))
        mape_list.append(mape)

        if show_plot:
            plt.figure(1, figsize=(20, 5))
            plt.plot([k + seq_len + num_obs_to_train - seq_len \
                      for k in range(seq_len)], p50, "r-")
            plt.fill_between(x=[k + seq_len + num_obs_to_train - seq_len for k in range(seq_len)], \
                             y1=p10, y2=p90, alpha=0.5)
            plt.title('Prediction uncertainty')
            yplot = yte[-1, -seq_len - num_obs_to_train:]
            plt.plot(range(len(yplot)), yplot, "k-")
            plt.legend(["P50 forecast", "true", "P10-P90 quantile"], loc="upper left")
            ymin, ymax = plt.ylim()
            plt.vlines(seq_len + num_obs_to_train - seq_len, ymin, ymax, color="blue", linestyles="dashed", linewidth=2)
            plt.ylim(0, ymax)
            plt.xlabel("Periods")
            plt.ylabel("Y")
            plt.show()
