datas = []
with open("data_LSTM.txt", "r+", encoding="UTF-8") as f:
    arr_list = eval(f.read())
    print(len(arr_list))
    datas = arr_list
    f.close()

for index,val in enumerate(datas):
    print(val)
    break

