import numpy as np
from elm import  ELMRegressor,GenELMRegressor
from random_layer import RandomLayer, MLPRandomLayer, RBFRandomLayer, GRBFRandomLayer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error # 均方误差
import pandas as pd
from pandas import DataFrame
from pandas import concat
import math
import statsmodels.api as sm
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from math import sqrt
from statsmodels.stats.diagnostic import acorr_ljungbox
import scipy.interpolate 
from sklearn import preprocessing
from sklearn.utils import check_random_state
import time
from matplotlib import pyplot as plt
from matplotlib.pylab import mpl
import copy

mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
mpl.rcParams['axes.unicode_minus']=False       #显示负号

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	将时间序列重构为监督学习数据集.
	参数:
		data: 观测值序列，类型为列表或Numpy数组。
		n_in: 输入的滞后观测值(X)长度。
		n_out: 输出观测值(y)的长度。
		dropnan: 是否丢弃含有NaN值的行，类型为布尔值。
	返回值:
		经过重组后的Pandas DataFrame序列.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# 输入序列 (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# 预测序列 (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# 将列名和数据拼接在一起
	agg = concat(cols, axis=1)
	agg.columns = names
	# 丢弃含有NaN值的行
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def objfun_elm(bw,x_train, y_train,n_hidden):
    '''
    优化所用的损失函数，返回误差值。
    n_hidden 隐藏神经元数
    wb为长度为2的横向量，即ELM中参数w和b的值
    '''
    bias=bw[0]
    weights=bw[1]
    rhl = RandomLayer(n_hidden=n_hidden, alpha=1.0)#表示_use_mlp_input=true
    elmr = GenELMRegressor(hidden_layer=rhl)
    elmr.fit(x_train, y_train,bias=bias,weights=weights)
    score=elmr.score(x_train, y_train)
    return score

def _compute_biases(n_hidden,rs):
    """Generate  biases"""
    biases = np.random.randn(n_hidden)
    return biases

def _compute_weights(n_hidden, n_features,rs):
    """Generate  weights"""
    weights =np.random.randn(n_features, n_hidden)
    weights *= 3.0 / n_features**0.5  # high dimensionality fix
    return weights

# 初始化萤火虫位置
def init_ffa(n,d,Lb,Ub,x_train,y_train):
    time1=time.time()
    n_hidden=64 #隐藏节点数
    n_features=x_train.shape[1]
    ns = []
    Lightn=[]
    while len(ns)<n:
        temp3=[]    
        rs = check_random_state(None)
        temp1=_compute_biases(n_hidden,rs)
        temp2=_compute_weights(n_hidden,n_features,rs)
        temp3.append(temp1)
        temp3.append(temp2)
        score=objfun_elm(temp3,x_train,y_train,n_hidden)
        if score>=0 and score<=1:
            ns.append(temp3)
            Lightn.append(score)
    time2=time.time()
    print('init time:%d' %(time2-time1))   
    print(Lightn)
    return [ns,Lightn]

# 每只萤火虫向更亮的个体移动
def ffa_move(n,d,ns,Lightn,nso,Lighto,alpha,betamin,gamma,Lb,Ub,Lightbest,x_train, y_train):
    # 参数取值范围绝对值 Scaling of the system
    scale0=np.ones(ns[0][0].shape[0])*abs(Ub[0]-Lb[0])
    scale1=np.ones(ns[0][0].shape[0])*abs(Ub[1]-Lb[1])
    # 更新萤火虫
    for i in range(n):
        #最亮的萤火虫进行移动
        if(Lightn[i]==Lightbest):
            tmpf=alpha*(np.random.uniform(-0.5,0.5,size=ns[i][0].shape[0]))*scale0
            temp0=ns[i][0]+tmpf;#i向j移动
            temp1=copy.deepcopy(ns[i][1])
            for k in range(len(ns[i][1])):
                tmpf1=alpha*(np.random.uniform(-0.5,0.5,size=ns[i][0].shape[0]))*scale1;
                temp1[k]=ns[i][1][k]+tmpf1;
            temp2=[]
            temp2.append(temp0)
            temp2.append(temp1)
            score=objfun_elm(temp2,x_train, y_train,64)
            if abs(score-1)< abs(Lightn[i]-1):
                ns[i]=temp2
                Lightn[i]=score                  
        else: #其它萤火虫进行移动
        # 吸引度参数 beta=exp(-gamma*r)
          for j in range(n):
                #计算欧式距离
                summ=(ns[i][0]-ns[j][0])**2
                for k in range(len(ns[i][1])):
                    summ=(ns[i][1][k]-nso[j][1][k])**2+summ
                r=np.sqrt(sum(summ)) 
                # Update moves
                if abs(Lightn[i]-1)>abs(Lighto[j]-1):  #如果j比i亮度更强(更接近1) Brighter and more attractive
                    beta0=1;
                    beta=(beta0-betamin)*np.power(math.e,-gamma*np.power(r,2))+betamin;
                    tmpf=alpha*(np.random.uniform(-0.5,0.5,size=ns[i][0].shape[0]))*scale0
                    temp0=ns[i][0]+beta0*(ns[i][0]-nso[j][0])*math.e**(-gamma*np.power(r,2))+tmpf;#i向j移动
                    temp1=copy.deepcopy(ns[i][1])
                    for k in range(len(ns[i][1])):
                        tmpf1=alpha*(np.random.uniform(-0.5,0.5,size=ns[i][0].shape[0]))*scale1;
                        temp1[k]=ns[i][1][k]+beta0*(ns[i][1][k]-nso[j][1][k])*math.e**(-gamma*np.power(r,2))+tmpf1;
                    temp2=[]
                    temp2.append(temp0)
                    temp2.append(temp1)
                    score=objfun_elm(temp2,x_train, y_train,64)
                    if abs(score-1)< abs(Lightn[i]-1):
                        ns[i]=temp2
                        Lightn[i]=score
                        
                        
#     #                     else:
#     #                         print('not move:%s' %(score))
#     # 防止越界 
#     #return findlimits(n,ns,Lb,Ub);
#     #print(ns[0],ns[1][0])
    return ns,Lightn

# 这个函数是可选的，因为它不在原来的FA中。
# 减少随机性的想法是增加收敛性，但是，如果减少随机性太快，那么可能会过早收敛。所以要小心使用。
def alpha_new(alpha,Iter,MaxIter):
    #alpha=0.97**(400*Iter/MaxIter)
    alpha=alpha*0.97
    return alpha

# 防止越界 
def findlimits(n,ns,Lb,Ub):
    for i in range(len(ns)):
        for j in range(len(ns[i][0])):
            if ns[i][0][j]<Lb[0]:
                ns[i][0][j]=Lb[0]
            elif ns[i][0][j]>Ub[0]:
                ns[i][0][j]=Ub[0]
        for k in range(len(ns[i][1])):
            for l in range(len(ns[i][1][k])):
                if ns[i][0][j]<Lb[1]:
                    ns[i][1][k][l]=Lb[1]
                elif ns[i][0][j]>Ub[1]:
                    ns[i][1][k][l]=Ub[1]

    return ns


# ===萤火虫算法实现的开始 ======
#         Lb = lower bounds/limits
#         Ub = upper bounds/limits
#     para == optional (to control the Firefly algorithm)
# 输出: nbest   = 找到的最佳解决方案
#       fbest   = 最优目标值
#       NumEval = number of evaluations: n*MaxGeneration
# 可选:
#      alpha可以减少（以减少随机性）
# ---------------------------------------------------------
 
# 算法主程序开始 Start FA

def ffa_mincon_elm(d, Lb, Ub, para,train_wine,train_wine_labels):
    # ------------------------------------------------
    # n=萤火虫个数
    # MaxGeneration=最大迭代次数
    # alpha=0.25;      # 随机扰动的步长因子 0--1
    # betamn=0.20;     # 吸引度beta的最小值
    # gamma=1;         # 介质对光的吸收系数
    # ------------------------------------------------
    n=para[0];
    MaxGeneration=para[1];
    alpha=para[2];
    betamin=para[3];
    gamma=para[4];

    # 检查上界向量与下界向量长度是否相同 
    if len(Lb) !=len(Ub):
        print('Simple bounds/limits are improper!')
        return

    # 初始化目标函数值 Initial values of an array
    zn=np.ones((n,1))
    # ------------------------------------------------
    # 初始化萤火虫位置 generating the initial locations of n fireflies
    print('Init开始')
    print('features:%d' %(train_wine.shape[1]))
    [ns,Lightn]=init_ffa(n,d,Lb,Ub,train_wine,train_wine_labels);
    nbest=ns[0];
    Lightbest=Lightn[0];
    print('Init完成')
    time1=time.time()
    for k in range(MaxGeneration):# 迭代开始
        # 更新alpha（可选）This line of reducing alpha is optional
        alpha=alpha_new(alpha,k,MaxGeneration);

        # 找出当前最优 Find the current best
        nso=ns;
        Lighto=Lightn;
        for j in range(len(Lightn)):
            if(abs(Lightbest-1)>abs(Lightn[j]-1)):
                nbest=ns[j];
                Lightbest=Lightn[j];
        print('Lightbest %d:%f' %(k,Lightbest))
        
        #提前终止迭代以防止过拟合
        if(Lightbest>0.8):
            break
        
        # 向较优方向移动 Move all fireflies to the better locations
        ns,Lightn=ffa_move(n,d,ns,Lightn,nso,Lighto,alpha,betamin,gamma,Lb,Ub,Lightbest,train_wine,train_wine_labels);
    
    for j in range(len(Lightn)):
        if(abs(Lightbest-1)>abs(Lightn[j]-1)):
            nbest=ns[j];
            Lightbest=Lightn[j];
    print('Lightbest:%f' %(Lightbest))
    time2=time.time()
    print("迭代时间：%d秒" %(time2-time1))

    
    print(Lightn)
    
    return [nbest,Lightbest]


# # 外部调用的主函数
def FA_ELM(X_train,Y_train,n_hidden=64):
    ## FA优化参数
    # 参数向量 parameters [n N_iteration alpha betamin gamma]
    # n为种群规模，N_iteration为迭代次数
    time1=time.time()
    para=[20,50,0.3,0.2,1]

    #待优化参数上下界 Simple bounds/limits for d-dimensional problems
    d=2; # 待优化参数个数
    #weight(-1,1)  bias(0,1)
    Lb=np.array([-1,0]); # 下界
    Ub=np.array([1,1]); # 上界
    

    
     # 迭代寻优 Display results
    [bestsolutio,bestojb]=ffa_mincon_elm(d,Lb,Ub,para,X_train,Y_train);
    ## 打印参数选择结果
    bestw=bestsolutio[1];
    bestb=bestsolutio[0];
   

    #以最佳参数训练ELM
    rhl = RandomLayer(n_hidden=64, alpha=1.0)#表示_use_mlp_input=true
    elmr = GenELMRegressor(hidden_layer=rhl)
    elmr.fit(X_train, Y_train,bias=bestb,weights=bestw)    
    time2=time.time()
    useTime=time2-time1
    print("总用时：%d秒" %(useTime))
    print(bestw)
    print(bestb)
    return elmr


#计算指定置信水平下的预测准确率
#level为小数
def comput_acc(real,predict,level):
    num_error=0
    for i in range(len(real)):
        if abs(real[i]-predict[i])/real[i]>level:
            num_error+=1
    return 1-num_error/len(real)

