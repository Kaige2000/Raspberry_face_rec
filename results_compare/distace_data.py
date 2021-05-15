# coding: utf-8
import copy

import face_recognition
import os
import random
import numpy as np
from scipy.spatial.distance import pdist
import _tool


# 获取相同人脸的距离信息
# 返回值为字典
# [test_num, dectect_error, all_e_distance, all_m_distance, all_coslin, all_pccs]
def get_same_face_distance(filePath):
    # 获取目录下所有文件夹的名称
    file_names = os.listdir(filePath)
    test_num = 0
    dectect_error = 0
    all_e_distance = []
    all_m_distance = []
    all_coslin = []
    all_pccs = []

    # 遍历所有文件夹
    for file_name in file_names:
        # 获得图片路径
        image_path = filePath + '\\' + file_name
        image_names = os.listdir(image_path)

        # 随机选择选择文件夹内一张图片作为训练数据
        c = random.randint(0, len(image_names) - 1)
        print('NO.' + file_name + ' ' + image_names[c] + ' is used for train')
        # 将jpg文件加载到numpy 数组中

        image = face_recognition.load_image_file(
            image_path + '\\' + image_names[c])
        face_locations = face_recognition.face_locations(image)
        # 如果识别不到，调用GPU
        if len(face_locations) == 0:
            face_locations = face_recognition.face_locations(image, model="cnn")
            print("cpu run error, use GPU")
            dectect_error += 1

        face_encoding_train = face_recognition.face_encodings(image, face_locations)

        # 删除训练文件
        image_names.pop(c)

        one_e_distance = 0
        one_m_distance = 0
        one_pccs = 0
        one_Cosline = 0
        one_num = 0

        for image_name in image_names:
            image_test = face_recognition.load_image_file(image_path + '\\' + image_name)
            face_locations = face_recognition.face_locations(image_test)

            if len(face_locations) == 0:
                face_locations = face_recognition.face_locations(image_test, model="cnn")
                print("cpu run error, use GPU")
                dectect_error += 1

            face_encoding_test = face_recognition.face_encodings(image_test, face_locations)[0]
            # print(face_encoding_test)
            # print(face_encoding_train)

            # 计算欧式距离并累加
            e_distance = face_recognition.face_distance(face_encoding_train, face_encoding_test)
            all_e_distance.append(e_distance)
            one_e_distance += e_distance

            # 计算曼哈顿距离并累加
            m_distance = np.sum(np.abs(face_encoding_train[0] - face_encoding_test))
            # print(m_distance)
            all_m_distance.append(m_distance)
            one_m_distance += m_distance

            # 马氏距离，样本数大于维数，无法满足实际要求

            # 计算余弦相似度并累加
            Cosine = 1 - pdist(np.vstack([face_encoding_train[0], face_encoding_test]), 'cosine')[0]
            all_coslin.append(Cosine)
            one_Cosline += Cosine

            # 计算皮尔逊相关系数
            pccs = np.corrcoef([face_encoding_train[0], face_encoding_test])[1][0]
            all_pccs.append(pccs)
            one_pccs += pccs

            # print(" Name: " + file_name + " NO. " + image_name + ' e_distance: ' + str(e_distance))
            # print(" Name: " + file_name + " NO. " + image_name + ' m_distance: ' + str(m_distance))
            # print(" Name: " + file_name + " NO. " + image_name + ' coslin: ' + str(Cosine))
            # print(" Name: " + file_name + " NO. " + image_name + ' pccs ' + str(pccs))

            test_num += 1
            one_num += 1

        print("NO." + file_name + " is done ")
        # print("NO." + file_name + " average Euclidean distance is " + str(one_e_distance / one_num))/
        # print("NO." + file_name + " average Manhattan distance is " + str(one_m_distance / one_num))
        # print("NO." + file_name + " average coslin is " + str(one_Cosline / one_num))
        # print("NO." + file_name + " average pccs is " + str(one_pccs / one_num))

    # dectect_error_rate = f'{dectect_error / test_num:.3%}'

    return {'test_num': test_num, 'dectect_error':dectect_error, 'all_e_distance': all_e_distance,
            'all_m_distance': all_m_distance, 'all_coslin': all_coslin, 'all_pccs': all_pccs}


# 获取不同人脸的距离信息
# 返回值为字典
# [test_num, dectect_error, all_e_distance, all_m_distance, all_coslin, all_pccs]
def get_different_face_distance(filePath, rate):
    # 计算索引值
    i = 0
    all_e_distance = []
    all_m_distance = []
    all_Cosline = []
    all_pccs = []

    # 获得待测样本
    sample_file_names, known_face_encodings = _tool.get_known_name_encodings(filePath, rate)

    # 待测样本数（自身不发生比较）
    num = len(sample_file_names) - 1

    # 总测试数和检测错误数
    test_num = 0
    dectect_error = 0

    for sample_file_name in sample_file_names:
        # 获取样本人物中的图片
        image_path = filePath + '\\' + sample_file_name
        image_names = os.listdir(image_path)
        one_e_distance = 0
        one_m_distance = 0
        one_Cosline = 0
        one_pccs = 0

        # 对比其他待测样本与其的距离关系
        for image_name in image_names:
            # 深拷贝，指向不同内存
            compare_face_encodings = copy.deepcopy(known_face_encodings)
            compare_face_encodings.pop(i)
            # print(compare_face_encodings)
            image_test = face_recognition.load_image_file(image_path + '\\' + image_name)
            face_locations = face_recognition.face_locations(image_test)
            if len(face_locations) == 0:
                face_locations = face_recognition.face_locations(image_test, model="cnn")
                print("cpu run error, use GPU")
                dectect_error += 1
            test_num += 1
            face_encoding_test = face_recognition.face_encodings(image_test, face_locations)[0]

            one_e_distance += sum(face_recognition.face_distance(compare_face_encodings, face_encoding_test)) / num
            # print(_tool.m_face_distance(compare_face_encodings, face_encoding_test))
            one_m_distance += sum(_tool.m_face_distance(compare_face_encodings, face_encoding_test)) / num
            one_Cosline += sum(_tool.cos_face_distance(compare_face_encodings, face_encoding_test)) / num
            one_pccs += sum(_tool.p_face_distance(compare_face_encodings, face_encoding_test)) / num

        all_e_distance.append(one_e_distance/len(image_names))
        all_m_distance.append(one_m_distance/len(image_names))
        all_Cosline.append(one_Cosline/len(image_names))
        all_pccs.append(one_pccs/len(image_names))

        i += 1
        print(sample_file_name + ' is done' + ' average e distance to others is %f' % (one_e_distance/len(image_names)))
        print(sample_file_name + ' is done' + ' average m distance to others is %f' % (one_m_distance / len(image_names)))
        print(sample_file_name + ' is done' + ' average cosline to others %f' % (one_Cosline / len(image_names)))
        print(sample_file_name + ' is done' + ' average prrc is to others %f' % (one_pccs / len(image_names)))

    return {'test_num': test_num, 'dectect_error': dectect_error, 'all_e_distance': all_e_distance,
            'all_m_distance': all_m_distance, 'all_coslin': all_Cosline, 'all_pccs': all_pccs}


# 计算数组的最大值、最小值、平均值、中位数、标准差
def calculate(list1, name):
    _max = max(list1)
    _min = min(list1)
    _average = np.average(list1)
    _median = np.median(list1)
    s_de = np.std(list1, ddof=1)
    print('for ' + name + ': ')
    print(' max is %f, min is %f, average is %f, median is %f, Standard deviation is %f' % (
    _max, _min, _average, _median, s_de))


# 打印结果
def print_result(dir):
    for key, value in dir.items():
        if isinstance(value, list):
            calculate(value, key)


att_Path = 'C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\att-database-of-faces'
cas_Path = 'C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\CASIA-FaceV5'

# same_dict = get_same_face_distance(att_Path)
different_dict = get_different_face_distance(att_Path, 0.3)

# print_result(same_dict)
print_result(different_dict)
# samples = _tool.select_sample('C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\att-database-of-faces', 0.5)


