# AIops-Learning-and-Exploration
AIops-Learning and Exploration 
------------------------------------------------------------------------------------------------------------------------------------------------------------------
TY
时间序列预测
TY_FA-ELM_时间序列预测
  包含三个.py文件:elm.py、FA-ELM.py、random_layer.py。

  其中，FA-ELM.py为外部调用库，主要包含三个方法：FA_ELM()，series_to_supervised()，comput_acc()。

  调用FA_ELM(X_train,Y_train,n_hidden=64)方法生成一个FA-ELM模型，该方法内部将自动定义模型、调用萤火虫算法寻找最优参数、使用最优参数构建FA-ELM模型。

  调用series_to_supervised(data, n_in=1, n_out=1, dropnan=True)方法将时间序列重构为监督学习数据集。

  调用comput_acc(real,predict,level)计算指定置信水平下的预测准确率。

  elm.py为原生ELM代码，我们将其中随机生成参数的代码修改为手动传入的方式。

  random_layer.py为elm.py的依赖库，包含RandomLayer、BaseRandomLayer、GRBFRandomLayer、RBFRandomLayer、MLPRandomLayer。

TY_LSTM_时间序列预测
主要流程为：

导入数据
数据预处理
取每小时均值
取每天上午9点到晚上8点数据
滞后扩充数据
特征选择
二折划分数据集并标准化
数据shuffle
构建模型
训练模型
预测
误差评估
TY_TCN_时间序列预测
主要流程为：

导入数据
数据预处理
二折划分数据集并标准化
构建模型
训练模型
预测
误差评估
TY_PCA与ICA_降维对比
主要流程为：

导入数据
数据标准化
分别使用PCA与ICA进行降维
二折划分数据集
分别使用PCA与ICA降维后的数据构建三层MLP神经网络模型
训练模型
预测
模型对比评估
TY_TDA_时间序列预测
  该文件显示了如何使用giotto-tda创建时间序列预测任务的拓扑特征，以及如何将其集成到scikit-learn兼容的管道中。
  特别是，我们将专注于从数据上连续滑动的窗口创建的拓扑特征。在滑动窗口模型中，形状为（n_timestamps，n_features）的单个时间序列数组X变为数据上具有新形状（n_windows，n_samples_per_window，n_features）的窗口的时间序列。使用滑动窗口构建预测模型时会出现两个主要问题：

n_windows小于n_timestamps。这是因为我们不能在没有填充X的情况下拥有超过时间戳的窗口，并且giotto-tda不能做到这一点。如果我们决定在连续的窗口之间选择较大的跨度，则n_timestamps-n_windows会更大。
目标变量y需要与每个窗口适当地“对齐”，以便预测问题是有意义的，例如。我们不会从未来“泄漏”信息。特别是，需要对y进行“重新采样”，使其长度也为n_windows。 为了解决这些问题，giotto-tda提供了带有resample，transform_resample和fit_transform_resample方法的转换器选择。这些是从TransformerResamplerMixin基类继承的。此外，giotto-tda替代了scikit-learn的Pipeline，从而扩展了它，以允许将TransformerResamplerMixins与常规的scikit-learn估计器链接在一起。
TY_傅里叶变换与自相关分析_周期性检测
  该文件展示了在时间序列数据上检测周期性的流程，其大致流程为：傅里叶变换获取候选周期-->自相关图获取真周期。
具体流程为：
导入数据；
打印原始数据时序图查看数据中有无名明显的周期特性；
傅里叶变换获取候选周期：
快速傅里叶变换获取未归一化的双边频谱；
归一化处理获取双边频谱；
FFT具有对称性，所以取前半部分获得单边频谱；
打印功率谱密度图，获取候选周期。
打印不同滞后阶数的自相关图，确定最终周期。
-------------------------------------------------------------------------------------------------------------------------------------------------------
ZC
时间序列预测
预测模型主要包括AE_LSTM、AGE_LSTM、ARIMA、DeepAR、GRU、LSTM。将这些模型应用到电网主机服务器运行数据的预测中。

数据
数据集位于"/SSH/data",包括原始数据和处理后的服务器性能数据,前者时间间隔为5min,后者时间间隔为1小时



环境要求
本实验硬件环境如下所示,代码大部分使用GPU运行
  CPU：Intel(R) Core(TM) i9-10940X CPU @ 3.30GHz; GPU：GeForce RTX 3090; NVIDIA-SMI 455.23.04; Driver Version: 455.23.04; CUDA Version: 11.1
  pip3 install -r requirements.txt
  运行
  直接运行模型文件即可,例如: python3 AE_LSTM_MemoryPrediction_Model_CPU.py
  分析
  每个模型文件中存在一个Analysis.py,用于保存和分析模型参数以及预测结果
---------------------------------------------------------------------------------------------------------------------------------------------------------
XY
#时间序列异常检测 将胶囊网络应用到时间序列的异常检测中。 #数据 数据集是MINIST ##使用数据集 安装和下载数据集： pip install -r requirements.txt python download_data.py 开始训练： python main.py ##测试精度 python main.py --is_training False
