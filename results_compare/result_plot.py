# coding: utf-8
import accuracy_compare
import matplotlib.pyplot as plt
import numpy as np
import _tool
import time
from multiprocessing import Process
from multiprocessing import Pool

# 用来正常显示中文标签和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

parameter = []
att_accuracy_same = []
att_accuracy_all = []
cas_accuracy_same = []
cas_accuracy_all = []

att_Path = 'C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\att-database-of-faces'
cas_Path = 'C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\CASIA-FaceV5'

print('s')

for t in np.arange(0.92, 1, 0.002):
    # e 代表计算欧氏距离
    print(t)

    # att_sameFace_result = accuracy_compare.same_face_test(att_Path, 'p', t)
    att_allFace_result = accuracy_compare.all_face_test(att_Path, 0.6, 'p', t)
    # cas_sameFace_result = accuracy_compare.same_face_test(cas_Path, 'p', t)
    # cas_allFace_result = accuracy_compare.all_face_test(cas_Path, 0.6, 'p', t)

    parameter.append(t)
    # att_accuracy_same.append(att_sameFace_result.get('accuracy'))
    att_accuracy_all.append(att_allFace_result.get('accuracy'))
    # cas_accuracy_same.append(cas_sameFace_result.get('accuracy'))
    # cas_accuracy_all.append(cas_allFace_result.get('accuracy'))
#
# _tool.to_csv(parameter, 'same_result.csv')
# _tool.to_csv(att_accuracy_same, 'same_result.csv')
# _tool.to_csv(cas_accuracy_same, 'same_result.csv')

_tool.to_csv(parameter, 'all_result.csv')
_tool.to_csv(att_accuracy_all, 'all_result.csv')
# _tool.to_csv(cas_accuracy_all, 'all_result.csv')



# plt.plot(parameter, att_accuracy_same, color="r", linestyle="--", marker="*", linewidth=1.0, label='att-same')
# plt.plot(parameter, att_accuracy_all, color="r", linestyle="-", marker="o", linewidth=1.0, label='att-all')
# plt.plot(parameter, cas_accuracy_same, color="b", linestyle="--", marker="x", linewidth=1.0, label='cas-same')
# plt.plot(parameter, cas_accuracy_all, color="b", linestyle="-", marker=">", linewidth=1.0, label='cas-all')


# plt.legend(loc='upper left')
#
# plt.xlabel("参数取值")
# plt.ylabel("准确率")
#
# plt.title("测试")
#
# plt.grid(color="k", linestyle=":")
#
# plt.show()
