# coding: utf-8
import face_recognition
import os
import random
import numpy as np
from scipy.spatial.distance import pdist

address = 'knn_test.csv'


def to_csv(list, address):
    with open(address, 'ab') as f:
        list_rows = [list]
        np.savetxt(f, list_rows, delimiter=",", fmt='% s')


def get_average_distance(filePath, t):
    # 获取目录下所有文件夹的名称
    file_names = os.listdir(filePath)
    test_num = 0
    dectect_error = 0
    all_e_distance, all_m_distance, all_coslin, all_pccs = 0, 0, 0, 0

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
        one_Cosline = 0
        one_pccs = 0
        one_num = 0
        Cosline = 0
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
            one_e_distance += e_distance

            # 计算曼哈顿距离并累加/ 结果差别非常小?
            m_distance = np.sum(np.abs(face_encoding_train[0] - face_encoding_test))
            one_m_distance += m_distance

            # 马氏距离，样本数大于维数，无法满足实际要求

            # 计算夹角余弦并累加？
            Cosline = pdist(np.vstack([face_encoding_train[0], face_encoding_test]), 'cosine')[0]
            one_Cosline += Cosline

            # 计算皮尔逊相关系数
            pccs = np.corrcoef([face_encoding_train[0], face_encoding_test])[1][0]
            one_pccs += pccs

            print(" Name: " + file_name + " NO. " + image_name + ' e_distance: ' + str(e_distance))
            print(" Name: " + file_name + " NO. " + image_name + ' m_distance: ' + str(m_distance))
            print(" Name: " + file_name + " NO. " + image_name + ' coslin: ' + str(Cosline))
            print(" Name: " + file_name + " NO. " + image_name + ' pccs ' + str(pccs))

            test_num += 1
            one_num += 1

        print("NO." + file_name + " average Euclidean distance is " + str(one_e_distance / one_num))
        print("NO." + file_name + " average Manhattan distance is " + str(one_m_distance / one_num))
        print("NO." + file_name + " average coslin is " + str(one_Cosline / one_num))
        print("NO." + file_name + " average pccs is " + str(one_pccs / one_num))

        all_e_distance += one_e_distance
        all_m_distance += one_m_distance
        all_coslin += one_Cosline
        all_pccs += one_pccs

    dectect_error_rate = f'{dectect_error / test_num:.3%}'
    average_e_distance = all_e_distance / test_num
    average_m_distance = all_m_distance / test_num
    average_coslin = all_coslin / test_num
    average_pccs = all_pccs / test_num

    return test_num, dectect_error, dectect_error_rate, average_e_distance, average_m_distance, average_coslin, average_pccs


Path = 'C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\att-database-of-faces'
# Path = 'C:\\Users\\Kaige\\Desktop\\CASPEALFaceDatabase'
test_num, dectect_error, dectect_error_rate, average_e_distance, average_m_distance, average_coslin, average_pccs = get_average_distance(Path, 0.6)
print(average_e_distance)
print(average_m_distance)
print(average_coslin)
print(average_pccs)
