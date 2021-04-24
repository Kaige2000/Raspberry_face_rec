# coding: utf-8
import numpy as np
import csv


# 将人脸数据写入csv文件
def to_csv(list, address):
    with open(address, 'ab') as f:
        list_rows = [list]
        np.savetxt(f, list_rows, delimiter=",", fmt='% s')


# 读取文件
def get_data(address):
    with open(address, 'r') as f:
        reader = csv.DictReader(f)
        datas = [row for row in reader]
    return datas


# 获得距离（欧式距离）
def get_distance(d1, d2):
    res = 0
    for i in range(2, 10):
        res += (float(d1.get(str(i))) - float(d2.get(str(i))))**2
    # for key in ("1", "2", "3", "4"):
    #     res += (float(d1[key]) - float(d2[key]))**2
    # for key in range(1, 4):
    #     res += (float(d1[str(key)]) - float(d2[str(key)])) ** 2
    return res ** 0.5


# data为待测数据, trains为训练数据
def knn(N, data, trains):
    # 计算距离
    res = [
        {"result": train['name'], "distance": get_distance(data, train)}
        for train in trains
    ]
    # for train in trains:
    #     print(train["1"])

    sorted(res, key=lambda item: item['distance'])
    res2 = res[0:N]

    # 计算加权平均
    # 分别代表识别结果
    result = {"Alice": 0, "Bob": 0}
    sum = 0
    for r in res2:
        sum += r['distance']

    for r in res2:
        print(r['result'])
        # 设置权重
        result[r['result']] += 1 - r['distance'] / sum
    print(result)
