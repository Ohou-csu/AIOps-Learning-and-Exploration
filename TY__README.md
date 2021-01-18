# 时间序列预测
## TY_FA-ELM_时间序列预测
<p>&emsp;&emsp;包含三个.py文件:elm.py、FA-ELM.py、random_layer.py。</p>
<p>&emsp;&emsp;其中，<b>FA-ELM.py</b>为外部调用库，主要包含三个方法：FA_ELM()，series_to_supervised()，comput_acc()。</p>
<p>&emsp;&emsp;调用<b>FA_ELM(X_train,Y_train,n_hidden=64)</b>方法生成一个FA-ELM模型，该方法内部将自动定义模型、调用萤火虫算法寻找最优参数、使用最优参数构建FA-ELM模型。
<p>&emsp;&emsp;调用<b>series_to_supervised(data, n_in=1, n_out=1, dropnan=True)</b>方法将时间序列重构为监督学习数据集。</p>
<p>&emsp;&emsp;调用<b>comput_acc(real,predict,level)</b>计算指定置信水平下的预测准确率。</p>
<p>&emsp;&emsp;<b>elm.py</b>为原生ELM代码，我们将其中随机生成参数的代码修改为手动传入的方式。</p>
<p>&emsp;&emsp;<b>random_layer.py</b>为elm.py的依赖库，包含RandomLayer、BaseRandomLayer、GRBFRandomLayer、RBFRandomLayer、MLPRandomLayer。</p>

## TY_LSTM_时间序列预测
主要流程为：
1. 导入数据  
2. 数据预处理  
    * 取每小时均值  
    * 取每天上午9点到晚上8点数据  
    * 滞后扩充数据  
3. 特征选择  
4. 二折划分数据集并标准化  
5. 数据shuffle  
6. 构建模型  
7. 训练模型  
8. 预测  
9. 误差评估

## TY_TCN_时间序列预测
主要流程为：
1. 导入数据  
2. 数据预处理   
4. 二折划分数据集并标准化  
6. 构建模型  
7. 训练模型  
8. 预测  
9. 误差评估

## TY_PCA与ICA_降维对比
主要流程为：
1. 导入数据   
4. 数据标准化  
5. 分别使用PCA与ICA进行降维
6. 二折划分数据集
6. 分别使用PCA与ICA降维后的数据构建三层MLP神经网络模型  
7. 训练模型  
8. 预测  
9. 模型对比评估

## TY_TDA_时间序列预测
&emsp;&emsp;该文件显示了如何使用giotto-tda创建时间序列预测任务的拓扑特征，以及如何将其集成到scikit-learn兼容的管道中。   
&emsp;&emsp;特别是，我们将专注于从数据上连续滑动的窗口创建的拓扑特征。在滑动窗口模型中，形状为（n_timestamps，n_features）的单个时间序列数组X变为数据上具有新形状（n_windows，n_samples_per_window，n_features）的窗口的时间序列。使用滑动窗口构建预测模型时会出现两个主要问题： 
1. n_windows小于n_timestamps。这是因为我们不能在没有填充X的情况下拥有超过时间戳的窗口，并且giotto-tda不能做到这一点。如果我们决定在连续的窗口之间选择较大的跨度，则n_timestamps-n_windows会更大。 
2. 目标变量y需要与每个窗口适当地“对齐”，以便预测问题是有意义的，例如。我们不会从未来“泄漏”信息。特别是，需要对y进行“重新采样”，使其长度也为n_windows。 
为了解决这些问题，giotto-tda提供了带有resample，transform_resample和fit_transform_resample方法的转换器选择。这些是从TransformerResamplerMixin基类继承的。此外，giotto-tda替代了scikit-learn的Pipeline，从而扩展了它，以允许将TransformerResamplerMixins与常规的scikit-learn估计器链接在一起。

## TY_傅里叶变换与自相关分析_周期性检测
&emsp;&emsp;该文件展示了在时间序列数据上检测周期性的流程，其大致流程为：傅里叶变换获取候选周期-->自相关图获取真周期。  
具体流程为：  
1. 导入数据；
2. 打印原始数据时序图查看数据中有无名明显的周期特性；  
3. 傅里叶变换获取候选周期：
    * 快速傅里叶变换获取未归一化的双边频谱；
    * 归一化处理获取双边频谱；
    * FFT具有对称性，所以取前半部分获得单边频谱；
    * 打印功率谱密度图，获取候选周期。
4. 打印不同滞后阶数的自相关图，确定最终周期。