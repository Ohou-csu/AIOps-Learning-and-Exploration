#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.pylab import mpl
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima_model import ARIMA
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense,Dropout
from keras.layers import LSTM
from sklearn import preprocessing
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import openpyxl
from keras import losses
import keras
from sklearn.metrics import mean_squared_error # 均方误差
import time
from ELM_new import comput_acc


# In[2]:


mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
mpl.rcParams['axes.unicode_minus']=False       #显示负号


# #  取数据

# In[3]:


data=pd.read_csv('dataset.csv',encoding ='gbk',parse_dates=['DATA_DT'])
Date=pd.to_datetime(data.DATA_DT.str.slice(0,12))
data['data'] = Date.map(lambda x: x.strftime('%Y-%m-%d'))
data['time'] = Date.map(lambda x: x.strftime('%H:%M:%S'))
datanew=data.set_index(Date)
series = pd.Series(datanew['接收流量'].values, index=datanew['DATA_DT'])
values = series.values
values = values.reshape((len(values), 1))


# # 取每小时均值

# In[4]:


list=[]
i=0
s=0.000
for x in values:
    s=x[0]+s
    i=i+1
    if(i==12):
        list.append(s/12)
        i=0
        s=0


# # 取每天上午9点到晚上8点

# In[5]:


list2=[]
i=0
for x in list[8:]:
    i=i+1
    if(i<13):
        list2.append(x)
    if(i==24):
        i=0
series = pd.Series(list2)


# # 滞后扩充数据

# In[6]:


dataframe1 = pd.DataFrame()
num_hour = 168
for i in range(num_hour,0,-1):
    dataframe1['t-'+str(i)] = series.shift(i)
dataframe1['t'] = series.values
dataframe2=dataframe1[168:]
dataframe2.index=range(len(dataframe2))
dataframe2


# # 特征选择

# In[7]:


dataframe3=dataframe2[['t','t-1','t-2','t-3','t-4','t-5','t-6','t-7','t-8','t-9','t-10','t-11','t-12']]
dataframe3['d-1-1']=(dataframe2['t-13']+dataframe2['t-14'])/2
dataframe3['d-1-2']=(dataframe2['t-15']+dataframe2['t-16'])/2
dataframe3['d-1-3']=(dataframe2['t-17']+dataframe2['t-18'])/2
dataframe3['d-1-4']=(dataframe2['t-19']+dataframe2['t-20'])/2
dataframe3['d-1-5']=(dataframe2['t-21']+dataframe2['t-22'])/2
dataframe3['d-1-6']=(dataframe2['t-23']+dataframe2['t-24'])/2
dataframe3['d-2-1']=(dataframe2['t-25']+dataframe2['t-26']+dataframe2['t-27'])/3
dataframe3['d-2-2']=(dataframe2['t-28']+dataframe2['t-29']+dataframe2['t-30'])/3
dataframe3['d-2-3']=(dataframe2['t-31']+dataframe2['t-32']+dataframe2['t-33'])/3
dataframe3['d-2-4']=(dataframe2['t-34']+dataframe2['t-35']+dataframe2['t-36'])/3
dataframe3['d-3-1']=(dataframe2['t-37']+dataframe2['t-38']+dataframe2['t-39']+dataframe2['t-40'])/4
dataframe3['d-3-2']=(dataframe2['t-41']+dataframe2['t-42']+dataframe2['t-43']+dataframe2['t-44'])/4
dataframe3['d-3-3']=(dataframe2['t-45']+dataframe2['t-46']+dataframe2['t-47']+dataframe2['t-48'])/4
dataframe3['d-4']=(dataframe2['t-49']+dataframe2['t-50']+dataframe2['t-51']+dataframe2['t-52']+
                    dataframe2['t-53']+dataframe2['t-54']+dataframe2['t-55']+dataframe2['t-56']+
                    dataframe2['t-57']+dataframe2['t-58']+dataframe2['t-59']+dataframe2['t-60'])/12
dataframe3['d-5']=(dataframe2['t-61']+dataframe2['t-62']+dataframe2['t-63']+dataframe2['t-64']+
                    dataframe2['t-65']+dataframe2['t-66']+dataframe2['t-67']+dataframe2['t-68']+
                    dataframe2['t-69']+dataframe2['t-70']+dataframe2['t-71']+dataframe2['t-72'])/12
dataframe3['d-6']=(dataframe2['t-73']+dataframe2['t-74']+dataframe2['t-75']+dataframe2['t-76']+
                    dataframe2['t-77']+dataframe2['t-78']+dataframe2['t-79']+dataframe2['t-80']+
                    dataframe2['t-81']+dataframe2['t-82']+dataframe2['t-83']+dataframe2['t-84'])/12
dataframe3['d-7']=(dataframe2['t-85']+dataframe2['t-86']+dataframe2['t-87']+dataframe2['t-88']+
                    dataframe2['t-89']+dataframe2['t-90']+dataframe2['t-91']+dataframe2['t-92']+
                    dataframe2['t-93']+dataframe2['t-94']+dataframe2['t-95']+dataframe2['t-96'])/12
dataframe3['d-8']=(dataframe2['t-97']+dataframe2['t-98']+dataframe2['t-99']+dataframe2['t-100']+
                    dataframe2['t-101']+dataframe2['t-102']+dataframe2['t-103']+dataframe2['t-104']+
                    dataframe2['t-105']+dataframe2['t-106']+dataframe2['t-107']+dataframe2['t-108'])/12
dataframe3['d-9']=(dataframe2['t-109']+dataframe2['t-110']+dataframe2['t-111']+dataframe2['t-112']+
                    dataframe2['t-113']+dataframe2['t-114']+dataframe2['t-115']+dataframe2['t-116']+
                    dataframe2['t-117']+dataframe2['t-118']+dataframe2['t-119']+dataframe2['t-120'])/12
dataframe3['d-10']=(dataframe2['t-121']+dataframe2['t-122']+dataframe2['t-123']+dataframe2['t-124']+
                    dataframe2['t-125']+dataframe2['t-126']+dataframe2['t-127']+dataframe2['t-128']+
                    dataframe2['t-129']+dataframe2['t-130']+dataframe2['t-131']+dataframe2['t-132'])/12
dataframe3['d-11']=(dataframe2['t-133']+dataframe2['t-134']+dataframe2['t-135']+dataframe2['t-136']+
                    dataframe2['t-137']+dataframe2['t-138']+dataframe2['t-139']+dataframe2['t-140']+
                    dataframe2['t-141']+dataframe2['t-142']+dataframe2['t-143']+dataframe2['t-144'])/12
dataframe3['d-12']=(dataframe2['t-145']+dataframe2['t-146']+dataframe2['t-147']+dataframe2['t-148']+
                    dataframe2['t-149']+dataframe2['t-150']+dataframe2['t-151']+dataframe2['t-152']+
                    dataframe2['t-153']+dataframe2['t-154']+dataframe2['t-155']+dataframe2['t-156'])/12
dataframe3['d-13']=(dataframe2['t-157']+dataframe2['t-158']+dataframe2['t-159']+dataframe2['t-160']+
                    dataframe2['t-161']+dataframe2['t-162']+dataframe2['t-163']+dataframe2['t-164']+
                    dataframe2['t-165']+dataframe2['t-166']+dataframe2['t-167']+dataframe2['t-168'])/12


# In[8]:


dataframe3


# # 二折划分数据并标准化

# In[9]:


# pot=int(len(dataframe3)*0.8)
pd.DataFrame(np.random.shuffle(dataframe3.values))  #shuffle
pot=len(dataframe3)-12
train=dataframe3[:pot]
test=dataframe3[pot:]
scaler = MinMaxScaler(feature_range=(0, 1)).fit(train)
#scaler = preprocessing.StandardScaler().fit(train)
train_norm=pd.DataFrame(scaler.fit_transform(train))
test_norm=pd.DataFrame(scaler.transform(test))
train_norm.columns=['t','t-1','t-2','t-3','t-4','t-5','t-6','t-7','t-8','t-9','t-10','t-11','t-12',
                    'd-1-1','d-1-2','d-1-3','d-1-4','d-1-5','d-1-6','d-2-1','d-2-2','d-2-3','d-2-4',
                    'd-3-1','d-3-2','d-3-3','d-4','d-5','d-6','d-7','d-8','d-9','d-10','d-11','d-12','d-13']
test_norm.columns=['t','t-1','t-2','t-3','t-4','t-5','t-6','t-7','t-8','t-9','t-10','t-11','t-12',
                    'd-1-1','d-1-2','d-1-3','d-1-4','d-1-5','d-1-6','d-2-1','d-2-2','d-2-3','d-2-4',
                    'd-3-1','d-3-2','d-3-3','d-4','d-5','d-6','d-7','d-8','d-9','d-10','d-11','d-12','d-13']


# In[10]:


train_norm.shape


# In[11]:


train_norm=train_norm[6:] # 便于分banch
train_norm.shape


# In[12]:


test_norm.shape


# # 打乱数据

# In[13]:


# pd.DataFrame(np.random.shuffle(train_norm.values))  #shuffle
# train_norm


# In[14]:


X_train=train_norm.iloc[:,1:]
X_test=test_norm.iloc[:,1:]
Y_train=train_norm.iloc[:,:1]
Y_test=test_norm.iloc[:,:1]


# # reshape input to be 3D [samples, timesteps, features]

# In[15]:


source_x_train=X_train
source_x_test=X_test
X_train=X_train.values.reshape([X_train.shape[0],1,X_train.shape[1]]) #从(696, 35)-->(696, 1，35)
X_test=X_test.values.reshape([X_test.shape[0],1,X_test.shape[1]])  #从(174, 35)-->(174, 1,35)


# In[16]:


X_test.shape


# In[17]:


X_train.shape


# In[18]:


X_train


# # 动态调整学习率与提前终止函数

# In[19]:


def lr_schedule(epoch):
    """
    根据epoch来调整学习率
    :param epoch:
    :return:
    """
    learning_rate = 0.01
    if epoch%50==0:
        learning_rate=learning_rate/(10**(epoch/50))
    return learning_rate


# In[20]:


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr


# In[21]:


# 创建提前终止调整回调
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0.001,verbose=1, mode='auto')
# 创建学习率调整回调
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)


# # 构建模型

# In[22]:


# 特征数
input_size = X_train.shape[2]
# 时间步长：用多少个时间步的数据来预测下一个时刻的值
time_steps = X_train.shape[1]
# 隐藏层block的个数
cell_size = 128
batch_size=12

model = keras.Sequential()
#使用LSTM
model.add(keras.layers.LSTM(
        units = cell_size, # 输出维度
        batch_input_shape=(batch_size, time_steps, input_size),# 输入维度
        stateful=True #保持状态
))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
# 输出层
model.add(keras.layers.Dense(1))

# 定义优化器
adam = keras.optimizers.Adam(lr=1e-4)
lr_metric = get_lr_metric(adam)
model.compile(optimizer=adam, loss='mae', metrics=['accuracy',lr_metric])


# In[23]:


#model.summary()


# # 训练模型

# In[24]:


time1=time.time()
nb_epoch=50#500
learning_rate=1e-2
min_lr=1e-4
last_loss=10000.0 #存储上一次的loss值
patience=1 #达到10后结束循环
threshold=min_lr*0.5

for i in range(nb_epoch):
    print('这是第%d次迭代：patience=%s' %(i,patience))
    #我们希望网络在观察序列中学习时建立状态，所以通过将“ shuffle”设置为“ False”来禁用样本shuffle。
    history=model.fit(X_train, Y_train,epochs=1,batch_size=batch_size,shuffle=False)        
     #每跑完一个epoch重置状态
    model.reset_states()
    
    #提前终止
    loss=history.history['loss'][0]
    if  abs(loss-last_loss)<threshold:
        patience+=1
        if patience>=10:
            break
    else:
        patience=0
    last_loss=loss
  
    #每50个epoch学习率*0.1
    if i>0 and learning_rate*0.1>=min_lr and i%50==0:
        learning_rate*=0.1
        adam = keras.optimizers.Adam(lr=learning_rate)
        lr_metric = get_lr_metric(adam)
        model.compile(optimizer=adam, loss='mae', metrics=['accuracy',lr_metric])
        
time2=time.time()


# In[25]:


print(time2-time1)


# # 预测

# In[26]:


yhat = model.predict(X_test)


# In[27]:


round(mean_squared_error(Y_test,yhat),10)


# In[30]:


real_predict=scaler.inverse_transform(np.concatenate((source_x_test,yhat),axis=1))
real_y=scaler.inverse_transform(np.concatenate((source_x_test,Y_test),axis=1))
real_predict=real_predict[:,-1]
real_y=real_y[:,-1]
plt.figure(figsize=(15,6))
plt.plot(real_predict,label='real_predict')
plt.plot(real_y,label='real_y')
plt.legend()
plt.show()


# In[31]:


from sklearn.metrics import mean_squared_error # 均方误差
round(math.sqrt(mean_squared_error(real_predict,real_y)),1)


# In[32]:


from sklearn.metrics import r2_score
round(r2_score(real_y,real_predict),4)


# In[33]:


real_y.shape,real_predict.shape


# In[34]:


plt.figure(figsize=(15,6))
bwith = 0.75 #边框宽度设置为2
ax = plt.gca()#获取边框
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)
plt.plot(real_predict,label='real_predict')
plt.plot(real_y,label='real_y')
plt.plot(real_y*(1+0.15),label='15%上限',linestyle='--',color='green')
# plt.plot(real_y*(1+0.1),label='10%上限',linestyle='--')
# plt.plot(real_y*(1-0.1),label='10%下限',linestyle='--')
plt.plot(real_y*(1-0.15),label='15%下限',linestyle='--',color='green')
plt.fill_between(range(0,12),real_y*(1+0.15),real_y*(1-0.15),color='gray',alpha=0.2)
plt.legend()
plt.show()


# In[39]:


per_real_loss=(real_y-real_predict)/real_y
print(round(sum(abs(per_real_loss))/len(per_real_loss),4))


# In[40]:


comput_acc(real_y,real_predict,0.2),comput_acc(real_y,real_predict,0.15),comput_acc(real_y,real_predict,0.1)


# In[41]:


plt.figure(figsize=(15,6))
plt.plot(per_real_loss,label='真实误差百分比')
plt.legend()
plt.show()


# In[ ]:




