# coding: utf-8
import face_recognition
import knn

Alice = []
Bob = []
address = 'knn_test.csv'

# 写入测试
# for i in range(1, 6):
#     Alice_image = face_recognition.load_image_file(
#         "C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\att-database-of-faces\\s2\\" + str(i) + ".png")
#     Alice_face_encoding = face_recognition.face_encodings(Alice_image)[0]
#     A = ['Alice']
#     A.extend(Alice_face_encoding)
#     Alice.append(A)
#
# for alice in Alice:
#     knn.to_csv(alice, address)
#
# for i in range(1, 11):
#     Bob_image = face_recognition.load_image_file(
#         "C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\att-database-of-faces\\s3\\" + str(i) + ".png")
#     Bob_face_encoding = face_recognition.face_encodings(Bob_image)[0]
#     B = ['Bob']
#     B.extend(Bob_face_encoding)
#     Bob.append(B)
# for bob in Bob:
#      knn.to_csv(bob, address)



# a = [7, 13, 5, 4]
# address = 'C:\\Users\\Kaige\\Desktop\\test.csv'
# knn_train.to_csv(a, address)
# address = 'knn_test'
# knn.to_csv(a, address)
#

datas = knn.get_data(address)
datas_test = datas[0:2]
datas_train = datas[3:]
knn.knn(3, datas_test[1], datas_train)
# print(a)
# print(datas_test)
# print(datas_train)
# list1 = []
# for data_test in datas_test:
#      list1.append(data_test.get('name'))
#
# print(datas_train)
# print(list1)
# #

# dates_test[0].get(2)

# a = float(datas_test[1].get('10'))
# b = float(datas_train[5].get('10'))

# print(a, b)
# print(datas_test)
# print(datas_train)
