# coding: utf-8
# 工具包
# coding: utf-8
import csv
import os
import random

import face_recognition
import numpy as np
from scipy.spatial.distance import pdist


def result_print(test_num, detect_error, detect_error_rate, identify_error, identify_error_rate):
    print("same face test start")
    print('the num of sample is %d' % test_num)
    print('detect error num is %d, detect error rate is %s' % (detect_error, detect_error_rate))
    print('identify error num is %d, identify error rate is %s' % (identify_error, identify_error_rate))


# 计算欧式距离并返回值
def face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    e_list = np.linalg.norm(face_encodings - face_to_compare, axis=1)

    return e_list


# 通过比较欧式距离实现人脸匹配，返回比较列表
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    return list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


# 计算曼哈顿距离并返回值
def m_face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    All_Manhattan = []
    for face_encoding in face_encodings:
        m_distance = np.sum(np.abs(face_encoding - face_to_compare))
        # if m_distance != 0:
        All_Manhattan.append(m_distance)
    return np.array(All_Manhattan)


# 通过比较曼哈顿距离实现人脸匹配，返回比较列表
def m_compare_faces(known_face_encodings, face_encoding_to_check, tolerance=3):
    return list(m_face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)


# 计算余弦距离并返回值
def cos_face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    All_Cosine = []
    for face_encoding in face_encodings:
        Cosine = 1 - pdist(np.vstack([face_encoding, face_to_compare]), 'cosine')
        if Cosine[0] != 1:
            All_Cosine.append(Cosine[0])
    return np.array(All_Cosine)


# 通过比较余弦距离实现人脸匹配，返回比较列表
def cos_compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.96):
    return list(cos_face_distance(known_face_encodings, face_encoding_to_check) >= tolerance)


# 计算皮尔逊系数并返回值
def p_face_distance(face_encodings, face_to_compare):
    if len(face_encodings) == 0:
        return np.empty((0))
    All_pccs = []
    for face_encoding in face_encodings:
        pccs = np.corrcoef([face_encoding, face_to_compare])[1][0]
        # if pccs != 0:
        All_pccs.append(pccs)
    return np.array(All_pccs)


# 通过比较皮尔逊系数实现人脸匹配，返回比较列表
def p_compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.97):
    return list(p_face_distance(known_face_encodings, face_encoding_to_check) >= tolerance)


def face_difference(face_encodings, face_to_compare, method):
    if len(face_encodings) == 0:
        return np.empty(0)
    if method == 'e':
        return face_distance(face_encodings, face_to_compare)
    elif method == 'm':
        return m_face_distance(face_encodings, face_to_compare)
    elif method == 'c':
        return cos_face_distance(face_encodings, face_to_compare)
    elif method == 'p':
        return p_face_distance(face_encodings, face_to_compare)


# 通过不同的方式比较人脸，返回真值列表
def compare_face(known_face_encodings, face_encoding_to_check, tolerance, method):
    if method == 'e' or method == 'm':
        # print(face_difference(known_face_encodings, face_encoding_to_check, method))
        # b = np.delete(face_difference(known_face_encodings, face_encoding_to_check, method), [0])
        b = face_difference(known_face_encodings, face_encoding_to_check, method)
        print(b)
        return list(b <= tolerance)
    elif method == 'c' or method == 'p':
        print(face_difference(known_face_encodings, face_encoding_to_check, method))
        return list(face_difference(known_face_encodings, face_encoding_to_check, method) >= tolerance)


# 获取一定量的样本
def select_sample(filePath, rate):
    file_names = os.listdir(filePath)
    samples = random.sample(file_names, int(len(file_names)*rate))
    return samples


#  获取数据文件中的姓名和面部编码
def get_known_name_encodings(filePath, rate):
    known_face_encodings = []
    sample_file_names = select_sample(filePath, rate)
    for sample_file_name in sample_file_names:
        # 获得图片路径
        image_path = filePath + '\\' + sample_file_name
        image_names = os.listdir(image_path)
        # 随机选择选择文件夹内一张图片作为训练数据
        c = random.randint(0, len(image_names) - 1)
        image = face_recognition.load_image_file(
            image_path + '\\' + image_names[c])
        face_locations = face_recognition.face_locations(image)
        # 如果识别不到，调用GPU
        if len(face_locations) == 0:
            face_locations = face_recognition.face_locations(image, model="cnn")
            print("cpu run error, use GPU")
        known_face_encodings.extend(face_recognition.face_encodings(image, face_locations))
        print('%s is registered, in which %s is for train' % (sample_file_name, image_names[c]))
    return sample_file_names, known_face_encodings


# 写入scv文件
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


# 读取文件
def get_data(address):
    with open(address, 'r') as f:
        reader = csv.DictReader(f)
        datas = [row for row in reader]
    return datas

#
# # 输入数据为字典，增加识可信度
# def all_face_test(train_data, test_data):
#     for keytrain_data