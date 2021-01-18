#!/usr/bin/env python
# coding: utf-8

# In[2]:


# https://datamarket.com/data/set/22ox/monthly-milk-production-pounds-per-cow-jan-62-dec-75#!ds=22ox&display=line
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tcn import TCN


# # 获取数据

# In[2]:


lookback_window = 12
#取数据
data=pd.read_csv('dataset.csv',encoding ='gbk',parse_dates=['DATA_DT'])
Date=pd.to_datetime(data.DATA_DT.str.slice(0,12))
data['data'] = Date.map(lambda x: x.strftime('%Y-%m-%d'))
data['time'] = Date.map(lambda x: x.strftime('%H:%M:%S'))
datanew=data.set_index(Date)
series = pd.Series(datanew['接收流量'].values, index=datanew['DATA_DT'])
values = series.values[:300]
values = values.reshape((len(values), 1))


# In[3]:


pot=300-12-lookback_window
train=values[:pot]
test=values[pot:300]


# In[4]:


x, y = [], []
for i in range(lookback_window, len(train)):
    x.append(train[i - lookback_window:i])
    y.append(train[i])
x = np.array(x)
y = np.array(y)

print(x.shape)
print(y.shape)


# In[5]:


x_test, y_test = [], []
for i in range(lookback_window, len(test)):
    x_test.append(test[i - lookback_window:i])
    y_test.append(test[i])
x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_test.shape)
print(y_test.shape)


# # 构建模型

# In[6]:


i = Input(shape=(lookback_window, 1))
m = TCN()(i)
m = Dense(1, activation='linear')(m)

model = Model(inputs=[i], outputs=[m])

model.summary()


# In[7]:


model.compile('adam', 'mae')
print('Train...')
model.fit(x, y, epochs=100, verbose=2,batch_size=64)


# # 误差评估

# In[8]:


p = model.predict(x_test)
plt.figure(figsize=(15,6))
plt.plot(p)
plt.plot(y_test)
plt.title('Monthly Milk Production (in pounds)')
plt.legend(['predicted', 'actual'])
plt.show()


# In[9]:


from sklearn.metrics import r2_score
round(r2_score(p,y_test),4)


# In[10]:


real_y=y_test
real_predict=p
per_real_loss=(real_y-real_predict)/real_y
avg_per_real_loss=sum(abs(per_real_loss))/len(per_real_loss)
print(avg_per_real_loss)


# In[ ]:




