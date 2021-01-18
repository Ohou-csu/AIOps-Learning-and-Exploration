#时间序列异常检测
将胶囊网络应用到时间序列的异常检测中。
#数据
数据集是MINIST
##使用数据集
安装和下载数据集：
pip install -r requirements.txt
python download_data.py
开始训练：
python main.py
##测试精度
python main.py --is_training False
