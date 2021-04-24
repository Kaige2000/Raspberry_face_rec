# -*- coding: utf-8 -*-

import face_recognition
import numpy as np
import joblib
from sklearn import svm
# 1 vs. 1的投票机制
# (C=1, kernel='rbf', probability=True, gamma=2)
'''
参数设置
C：C-SVC的惩罚参数C?默认值是1.0
C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
kernel：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
0 – 线性：u'v
1 – 多项式：(gamma*u'*v + coef0)^degree
2 – RBF函数：exp(-gamma|u-v|^2)
3 –sigmoid：tanh(gamma*u'*v + coef0)
degree：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。
gamma：‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features
coef0：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。
probability：是否采用概率估计？.默认为False

'''

# X = [[0, 0], [1, 1]]
# y = [0, 1]
# clf = svm.SVC()
# clf.fit(X, y)
#
# joblib.dump(clf, "train_model.m")

# 训练输入
# Alice_image = face_recognition.load_image_file("C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\att-database-of-faces\\s1\\1.png")
# Bob_image = face_recognition.load_image_file("C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\att-database-of-faces\\s2\\1.png")
# Alice_face_encoding = face_recognition.face_encodings(Alice_image)[0]
# Bob_face_encoding = face_recognition.face_encodings(Bob_image)[0]
# myface = Alice_face_encoding
# unkown = Bob_face_encoding
# # print(myface
# train = [myface, unkown]
# result = ["Alice", "Bob"]
# clf = svm.SVC()
# clf.fit(train, result)
# joblib.dump(clf, "train_test2.m")

# 测试输入
test_image = face_recognition.load_image_file("C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\att-database-of-faces\\s2\\4.png")
test_face_encoding = face_recognition.face_encodings(test_image)[0]

# print(test_face_encoding)
coding = test_face_encoding

clf = joblib.load('train_test2.m')
result = clf.predict([coding])
print(result)
