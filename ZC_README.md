<h1>时间序列预测</h1>
<p>预测模型主要包括AE_LSTM、AGE_LSTM、ARIMA、DeepAR、GRU、LSTM。将这些模型应用到电网主机服务器运行数据的预测中。
</p>
<h3>数据</h3>
<p>数据集位于"/SSH/data",包括原始数据和处理后的服务器性能数据,前者时间间隔为5min,后者时间间隔为1小时</p>
<img src="https://github.com/Ohou-csu/AIOps-Learning-and-Exploration/tree/main/Images/服务器性能数据.png">
<h3>环境要求</h3>
<p>本实验硬件环境如下所示,代码大部分使用GPU运行</p>
<blockquote>
CPU：Intel(R) Core(TM) i9-10940X CPU @ 3.30GHz;
GPU：GeForce RTX 3090;
NVIDIA-SMI 455.23.04;  
Driver Version: 455.23.04;
CUDA Version: 11.1
</blockquote>
<code>pip3 install -r requirements.txt</code>
<h3>运行</h3>
直接运行模型文件即可,例如:
<code>python3 AE_LSTM_MemoryPrediction_Model_CPU.py</code>
<h3>分析</h3>
每个模型文件中存在一个Analysis.py,用于保存和分析模型参数以及预测结果
<h3>参考</h3>






