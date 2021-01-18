import json
datas = []
with open("data_2_without_shuffle2.txt", "r+", encoding="UTF-8") as f:
    arr_list = eval(f.read())
    print(f"参数组个数：{len(arr_list)}")
    datas = arr_list
    f.close()

MAPEs = []
for index, val in enumerate(datas):
    # print(json.dumps(val))
    dict = val["Verification"]["MAPE_test"]
    MAPEs.append(dict)
    if index is MAPEs.index(min(MAPEs)):
        print(json.dumps(val))

