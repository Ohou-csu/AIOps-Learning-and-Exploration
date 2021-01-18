#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.decomposition import FastICA,PCA
from sklearn import preprocessing


# In[2]:


#取数据
datanew=pd.read_csv('dataset.csv',encoding ='gbk')


# In[3]:


#标准化
X=datanew[:1000]
data_normal = preprocessing.scale(X)


# In[4]:


#PAC降维
pca = PCA(n_components=5)
newX = pca.fit_transform(data_normal)     #等价于pca.fit(X) pca.transform(X)
#invX = pca.inverse_transform(X)  #将降维后的数据转换成原始数据
print(pca.explained_variance_ratio_)#输出降维后的各特征贡献度


# In[6]:


plt.figure(figsize=(12,8))
plt.title('PCA Components')
plt.scatter(newX[:,0], newX[:,1])
plt.scatter(newX[:,1], newX[:,2])
plt.scatter(newX[:,2], newX[:,3])
plt.scatter(newX[:,3], newX[:,4])
plt.scatter(newX[:,4], newX[:,0])


# In[7]:


trandata=newX[:900]
predictdata=newX[901:910]
Y=data_normal[:900,10]
model=tf.keras.Sequential([tf.keras.layers.Dense(10,input_shape=(5,),activation='relu'),
                           tf.keras.layers.Dense(10,activation='relu'),
                           tf.keras.layers.Dense(1)])
model.summary()#查看模型基本信息
model.compile(optimizer='adam',loss='mse',metrics=['mae']) #设置优化器和损失函数
history=model.fit(trandata,Y[:900],epochs=100) #每个数据训练1000次


# In[8]:


# ICA降维
ica = FastICA(n_components=5)
S_ = ica.fit_transform(data_normal)  # 重构信号
A_ = ica.mixing_  # 获得估计混合后的矩阵


# In[9]:


plt.figure(figsize=(12,8))
plt.title('ICA Components')
plt.scatter(S_[:,0], S_[:,1])
plt.scatter(S_[:,1], S_[:,2])
plt.scatter(S_[:,2], S_[:,3])
plt.scatter(S_[:,3], S_[:,4])
plt.scatter(S_[:,4], S_[:,0])


# In[10]:


trandata2=S_[:900]
model2=tf.keras.Sequential([tf.keras.layers.Dense(10,input_shape=(5,),activation='relu'),
                           tf.keras.layers.Dense(10,activation='relu'),
                           tf.keras.layers.Dense(1)])
model2.summary()#查看模型基本信息
model2.compile(optimizer='adam',loss='mse',metrics=['mae']) #设置优化器和损失函数
history2=model.fit(trandata2,Y[:900],epochs=100) #每个数据训练1000次


# In[11]:


#模型评估
model.evaluate(predictdata,data_normal[901:910,10])
model2.evaluate(predictdata,data_normal[901:910,10])


# In[13]:


plt.figure(figsize=(10,5))
history.history.keys() #查看history中存储了哪些参数
plt.subplot(1,2,1)
plt.title("PCA")
plt.plot(history.epoch,history.history.get('loss')) #画出随着epoch增大loss的变化图
history2.history.keys() #查看history中存储了哪些参数
plt.subplot(1,2,2)
plt.title("ICA")
plt.plot(history2.epoch,history2.history.get('loss')) #画出随着epoch增大loss的变化图
plt.show()


# In[ ]:




