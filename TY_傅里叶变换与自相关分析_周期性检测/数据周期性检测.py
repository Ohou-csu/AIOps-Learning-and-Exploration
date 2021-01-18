#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from matplotlib import pyplot as plt
from pandas import Series
from sklearn.preprocessing import StandardScaler
from math import sqrt
from statsmodels import api as sm
import scipy.io as sio
from scipy import fft, arange, signal
from matplotlib.pylab import mpl
from scipy import signal   #滤波等


# # 检测流程：傅里叶变换获取候选周期-->自相关图获取真周期

# In[2]:


mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
mpl.rcParams['axes.unicode_minus']=False       #显示负号


# # 取数据

# In[3]:


data=pd.read_csv('dataset.csv',encoding ='gbk',parse_dates=['DATA_DT'])
Date=pd.to_datetime(data.DATA_DT.str.slice(0,12))
data['data'] = Date.map(lambda x: x.strftime('%Y-%m-%d'))
data['time'] = Date.map(lambda x: x.strftime('%H:%M:%S'))
datanew=data.set_index(Date)
series = pd.Series(datanew['接收流量'].values, index=datanew['DATA_DT'])


# # 标准化

# In[4]:


# 准备要标准化的数据 
values = series.values
values = values.reshape((len(values), 1))

# 训练标准化规则 
scaler = StandardScaler()
scaler = scaler.fit(values)
print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))

# 标准化数据集并打印前5行 
normalized = scaler.transform(values)
# for i in range(5):
# 	print(normalized[i])


# # 3.转换为Series数据

# In[5]:


list=[]
i=0
for x in normalized:
    list.append(x[0])
    i=i+1
normalized_series = pd.Series(list, index=datanew['DATA_DT'])
data_temp = pd.Series(list)


# In[6]:


week_data=normalized_series['20200210':'20200216']
month_data=normalized_series['20200210':'20200310']


# # 4.查看原始数据时序图

# In[7]:


plt.figure(figsize = (15, 8))
plt.plot(pd.Series(list))
plt.show()


# In[8]:


plt.figure(figsize = (15, 8))
plt.plot(week_data.index,week_data.values)
plt.title('一周数据')
plt.show()


# ## 打印一周的数据能看到基本以天为周期

# In[9]:


plt.figure(figsize = (15, 8))
plt.plot(month_data.index,month_data.values)
plt.title('一月数据')
plt.show()


# ## 打印一月的数据能看出基本以周为周期

# # 5.傅里叶变换

# In[10]:


fft_y=fft(normalized_series)                          #快速傅里叶变换


# In[11]:


N=24942
x = np.arange(N)           # 频率个数
 
abs_y=np.abs(fft_y)                # 取复数的绝对值，即复数的模(双边频谱)
angle_y=np.angle(fft_y)              #取复数的角度
 
plt.figure(figsize = (15, 8))
plt.plot(x,abs_y)   
plt.title('双边振幅谱（未归一化）')
 
plt.figure(figsize = (15, 8))
plt.plot(x ,angle_y)   
plt.title('双边相位谱（未归一化）')
plt.show()


# ## 振幅值很大，进行归一化

# In[12]:


normalization_y=abs_y/N            #归一化处理（双边频谱）
plt.figure(figsize = (15, 8))
plt.plot(x,normalization_y,'g')
plt.title('双边频谱(归一化)',fontsize=9,color='green')
plt.show()


# ## FFT具有对称性，所以取前半部分即可

# In[13]:


half_x = x[range(int(N/2))]                                  #取一半区间
normalization_half_y = normalization_y[range(int(N/2))]      #由于对称性，只取一半区间（单边频谱）
plt.figure(figsize = (15, 8))
plt.plot(half_x,normalization_half_y,'b')
plt.title('单边频谱(归一化)',fontsize=9,color='blue')
plt.show()


# # 一天的功率谱密度图与自相关图对比

# In[14]:


# FFT
num_fft=600#len(normalized_series)
Y = fft(normalized_series[:600],num_fft )
Y = np.abs(Y)
plt.figure(figsize=(15, 9))
#plt.plot(20*np.log10(Y[:num_fft//2]))

# 功率谱图
ps = Y**2 / num_fft
plt.subplot(2,1,1)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.4)
tranformps=20*np.log10(ps[:num_fft//2])
plt.plot(tranformps)
plt.xlabel('频率')
plt.ylabel('power')
plt.title('功率谱密度图')
# 显示极值点
max_x1=250+np.argmax(tranformps[250:300])
max_y1=tranformps[max_x1]
plt.text(max_x1,max_y1 ,(max_x1,max_y1),ha='left', va='top', fontsize=13)
max_x2=270+np.argmax(ps[270:300])
max_y2=ps[max_x2]
plt.text(max_x2,max_y2 ,(max_x2,max_y2),ha='left', va='top', fontsize=13)

#互相关功率谱图
# cor_x = np.correlate(normalized_series[:600], normalized_series[:600], 'same')
# cor_X = fft(cor_x, num_fft)
# ps_cor = np.abs(cor_X)
# ps_cor = ps_cor / np.max(ps_cor)
#plt.plot(20*np.log10(ps_cor[:num_fft//2]))

acf = sm.tsa.acf(normalized_series, nlags=len(normalized_series))
lag = arange(len(normalized_series))
plt.subplot(2,1,2)
plt.plot(lag, acf)
plt.xlim((0, 300))
plt.xlabel('Lags (days)')
plt.ylabel('Autocorrelation')
plt.title('最大滞后一天的自相关图')
# 显示极值点
max_x1=250+np.argmax(acf[250:300])
max_y1=acf[max_x1]
plt.text(max_x1,max_y1 ,(max_x1,max_y1),ha='left', va='top', fontsize=13)


# ## 一天的候选周期为253，292，真实周期为287

# # 一周的功率谱密度图与自相关图对比

# In[15]:


# FFT
num_fft=288*7*2
Y = fft(normalized_series[:num_fft],num_fft )
Y = np.abs(Y)
plt.figure(figsize=(15, 9))

# 功率谱图
ps = Y**2 / num_fft
plt.subplot(2,1,1)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.4)
tranformps=20*np.log10(ps[:num_fft//2])
plt.plot(tranformps)
#plt.plot(ps//2)
plt.xlabel('频率')
plt.ylabel('power')
plt.title('功率谱密度图')
# 显示极值点
max_x1=1870+np.argmax(tranformps[1870:2015])
max_y1=tranformps[max_x1]
plt.text(max_x1,max_y1 ,(max_x1,max_y1),ha='left', va='top', fontsize=13)
max_x2=500+np.argmax(tranformps[500:750])
max_y2=tranformps[max_x2]
plt.text(max_x2,max_y2 ,(max_x2,max_y2),ha='left', va='top', fontsize=13)
max_x3=250+np.argmax(tranformps[250:400])
max_y3=tranformps[max_x3]
plt.text(max_x3,max_y3 ,(max_x3,max_y3),ha='left', va='top', fontsize=13)

#自相关图
plt.subplot(2,1,2)
plt.plot(lag, acf)
plt.xlim((0, 288*7))
plt.xlabel('Lags (days)')
plt.ylabel('Autocorrelation')
plt.title('最大滞后一周的自相关图')
# 显示极值点
max_x1=1870+np.argmax(acf[1870:2015])
max_y1=acf[max_x1]
plt.text(max_x1,max_y1 ,(max_x1,max_y1),ha='left', va='top', fontsize=13)
max_x2=500+np.argmax(acf[500:750])
max_y2=acf[max_x2]
plt.text(max_x2,max_y2 ,(max_x2,max_y2),ha='left', va='top', fontsize=13)
max_x3=250+np.argmax(acf[250:300])
max_y3=acf[max_x3]
plt.text(max_x3,max_y3 ,(max_x3,max_y3),ha='left', va='top', fontsize=13)
plt.show()


# In[16]:


plt.plot(20*np.log10(Y[:num_fft//2]))


# ## 一周为2016个点，周期为2003

# # 自相关分析

# In[17]:


series_len=len(normalized_series)
acf = sm.tsa.acf(normalized_series, nlags=series_len)
plt.figure(figsize = (15, 8))
lag = arange(series_len)
plt.plot(lag[:round(series_len/2)-3200], acf[:round(series_len/2)-3200])
plt.xlim((0, round(series_len/2)-3200))
plt.ylim((-0.25,0.6))
plt.xlabel('Lags')
plt.ylabel('Autocorrelation')
#plt.title('最大滞后整个序列长度')
# 显示极值点
max_x1=round(1000+np.argmax(acf[1000:3000]),4)
max_y1=round(acf[max_x1],4)
plt.text(max_x1,max_y1 ,(max_x1,max_y1),ha='left', va='top', fontsize=16)
max_x2=round(200+np.argmax(acf[200:1000]),4)
max_y2=round(acf[max_x2],4)
plt.text(max_x2,max_y2 ,(max_x2,max_y2),ha='left', va='top', fontsize=16)
max_x3=round(8000+np.argmax(acf[8000:9000]),4)
max_y3=round(acf[max_x3],4)
plt.text(max_x3,max_y3 ,(max_x3,max_y3),ha='left', va='top', fontsize=16)
max_x4=round(2500+np.argmax(acf[2500:5000]),4)
max_y4=round(acf[max_x4],4)
plt.text(max_x4,max_y4 ,(max_x4,max_y4),ha='left', va='top', fontsize=16)


# In[ ]:


acf = sm.tsa.acf(normalized_series, nlags=len(normalized_series))
plt.figure(figsize = (15, 8))
lag = arange(len(normalized_series))
plt.plot(lag, acf)
plt.xlim((0, 8650))
plt.xlabel('Lags (days)')
plt.ylabel('Autocorrelation')
plt.title('最大滞后一个月')
# 显示极值点
max_x1=round(1000+np.argmax(acf[1000:3000]),4)
max_y1=round(acf[max_x1],4)
plt.text(max_x1,max_y1 ,(max_x1,max_y1),ha='left', va='top', fontsize=13)
max_x2=round(200+np.argmax(acf[200:1000]),4)
max_y2=round(acf[max_x2],4)
plt.text(max_x2,max_y2 ,(max_x2,max_y2),ha='left', va='top', fontsize=13)
max_x3=round(8000+np.argmax(acf[8000:9000]),4)
max_y3=round(acf[max_x3],4)
plt.text(max_x3,max_y3 ,(max_x3,max_y3),ha='left', va='top', fontsize=13)
max_x4=round(2500+np.argmax(acf[2500:5000]),4)
max_y4=round(acf[max_x4],4)
plt.text(max_x4,max_y4 ,(max_x4,max_y4),ha='left', va='top', fontsize=13)


# In[ ]:


plt.figure(figsize = (15, 8))
lag = arange(len(normalized_series))
plt.plot(lag, acf)
plt.xlim((0, 1596))
plt.xlabel('Lags (days)')
plt.ylabel('Autocorrelation')
plt.title('最大滞后一周')
# 显示极值点
max_x1=500+np.argmax(acf[500:700])
max_y1=acf[max_x1]
plt.text(max_x1,max_y1 ,(max_x1,max_y1),ha='left', va='top', fontsize=13)
max_x2=200+np.argmax(acf[200:1000])
max_y2=acf[max_x2]
plt.text(max_x2,max_y2 ,(max_x2,max_y2),ha='left', va='top', fontsize=13)


# In[ ]:




