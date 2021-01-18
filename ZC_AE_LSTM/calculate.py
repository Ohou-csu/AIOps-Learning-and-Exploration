num_hour = 192  # 历史数据个数
input_size_AE = 24
hat_size_AE = 84

def allFactor(n):
    if n == 0: return [0]
    if n == 1: return [1]
    rlist = []
    for i in range(1,n+1):
        if n%i == 0:
            rlist.append(i)

    return rlist

list = allFactor(hat_size_AE)
print(list)

# input_size_lstm = int(hat_size_AE / (num_hour / 24))

for i in list:
    num_hour = (hat_size_AE / i) * 24
    if num_hour>=72 and num_hour<=672:
        print(f"input_size_lstm:{i}"+" "+ f"num_hour:{str(num_hour)}")

# input_size_lstm:3 num_hour:672.0
# input_size_lstm:4 num_hour:504.0
# input_size_lstm:6 num_hour:336.0
# input_size_lstm:7 num_hour:288.0
# input_size_lstm:12 num_hour:168.0
# input_size_lstm:14 num_hour:144.0
# input_size_lstm:21 num_hour:96.0
# input_size_lstm:28 num_hour:72.0