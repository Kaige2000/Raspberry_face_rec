# coding: utf-8
import tool
import knn_train

# a = [7, 13, 5, 4]
address = 'C:\\Users\\Kaige\\Desktop\\test.csv'
# knn_train.to_csv(a, address)

datas = knn_train.get_data(address)
datas_test = datas[0:2]
datas_train = datas[2:]

print(datas_test)

# list1 = []
# for data_test in datas_test:
#     list1.append(datas_test[1].get('1'))

# print(list1)

knn_train.knn(2, datas_test[1], datas_train)
