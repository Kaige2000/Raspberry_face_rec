# coding: utf-8
import face_recognition
import os
import random
import numpy as np
from scipy.spatial.distance import pdist


def get_all_distance(filePath):
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
        print(image_names[c] + ' is used for train')
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
        one_num = 0
        one_Cosline = 0

        for image_name in image_names:
            image_test = face_recognition.load_image_file(image_path + '\\' + image_name)
            face_locations = face_recognition.face_locations(image_test)

            if len(face_locations) == 0:
                face_locations = face_recognition.face_locations(image_test, model="cnn")
                print("cpu run error, use GPU")
                dectect_error += 1

            face_encoding_test = face_recognition.face_encodings(image_test, face_locations)[0]

            # 计算欧式距离并累加
            e_distance = face_recognition.face_distance(face_encoding_train, face_encoding_test)
            all_e_distance.append(e_distance)
            one_e_distance += e_distance

            # 计算曼哈顿距离并累加
            m_distance = np.sum(np.abs(face_encoding_train[0] - face_encoding_test))
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

            print(" Name: " + file_name + " NO. " + image_name + ' e_distance: ' + str(e_distance))
            print(" Name: " + file_name + " NO. " + image_name + ' m_distance: ' + str(m_distance))
            print(" Name: " + file_name + " NO. " + image_name + ' coslin: ' + str(Cosine))
            print(" Name: " + file_name + " NO. " + image_name + ' pccs ' + str(pccs))

            test_num += 1
            one_num += 1

        print("NO." + file_name + " is done ")
        print("NO." + file_name + " average Euclidean distance is " + str(one_e_distance / one_num))
        print("NO." + file_name + " average Manhattan distance is " + str(one_m_distance / one_num))
        print("NO." + file_name + " average coslin is " + str(one_Cosline / one_num))
        print("NO." + file_name + " average pccs is " + str(one_pccs / one_num))

    dectect_error_rate = f'{dectect_error / test_num:.3%}'

    return test_num, dectect_error, dectect_error_rate, all_e_distance, all_m_distance, all_coslin, all_pccs


def print_result(list, name):
    _max = max(list)
    _min = min(list)
    _average = np.average(list)
    _median = np.median(list)
    s_de = np.std(list, ddof=1)
    print('for ' + name + ': ')
    print(' max is %f, min is %f, average is %f, median is %f, Standard deviation is %f' % (
    _max, _min, _average, _median, s_de))


Path = 'C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\att-database-of-faces'
# Path = 'C:\\Users\\Kaige\\Desktop\\CASPEALFaceDatabase'

test_num, dectect_error, dectect_error_rate, all_e_distance, all_m_distance, all_coslin, all_pccs = get_all_distance(
    Path)

print('sort out......')
print_result(all_e_distance, 'Euclidean distance')
print_result(all_m_distance, 'Manhattan distance')
print_result(all_coslin, 'coslin')
print_result(all_pccs, 'Pearson correlation coefficient')

# average_e_distance = sum(all_e_distance) / test_num
# average_m_distance = sum(all_m_distance) / test_num
# average_coslin = sum(all_coslin) / test_num
# average_pccs = sum(all_pccs) / test_num

# print('average Euclidean distance is %f' % average_e_distance)
# print('average Manhattan Distance is %f' % average_m_distance)
# print('average coslin is %f' % average_coslin)
# print('average Pearson correlation coefficient is %f' % average_pccs)
