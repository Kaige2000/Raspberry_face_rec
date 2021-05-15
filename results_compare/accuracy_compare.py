# coding: utf-8
import copy

import face_recognition
import os
import random
import _tool


att_Path = 'C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\att-database-of-faces'
cas_Path = 'C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\CASIA-FaceV5'
# 输入文件夹父目录，返回测试样本总数，HOG\CPU人脸检测错误数，HOG\CPU人脸检测错误率，人脸识别错误数，人脸检测错误率
# 获取一定量的样本


# 相同人脸的识别结果
def same_face_test(filePath, method, t):
    # 获取目录下所有文件夹的名称
    file_names = os.listdir(filePath)
    test_num = 0
    detect_error = 0
    identify_error = 0

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
            detect_error += 1

        face_encoding_train = face_recognition.face_encodings(image, face_locations)

        # 删除用于训练的数据文件
        image_names.pop(c)

        # print(face_encoding_train)
        for image_name in image_names:
            image_test = face_recognition.load_image_file(image_path + '\\' + image_name)
            face_locations = face_recognition.face_locations(image_test)

            if len(face_locations) == 0:
                face_locations = face_recognition.face_locations(image_test, model="cnn")
                print("cpu run error, use GPU")
                detect_error += 1
            result = _tool.compare_face(face_encoding_train,
                                        face_recognition.face_encodings(image_test, face_locations)[0], t, method)
            if not result[0]:
                identify_error += 1
            print(" Name " + file_name + " NO. " + image_name + ' result: ' + str(result))
            test_num += 1
        print("NO." + file_name + " is done")

    # dectect_error_rate = f'{detect_error / test_num:.3%}'
    # indentify_error_rate = f'{identify_error / test_num:.3%}'
    # identify_error_rate = identify_error / test_num
    # accurcy = f'{1 - (detect_error + identify_error) / test_num:.3%}'
    accuracy = 1 - (detect_error + identify_error) / test_num
    return {'test_num': test_num, 'detect_error': detect_error, 'identify_error': identify_error,
            'accuracy': accuracy}


# 不同人脸的识别结果
def all_face_test(filePath, rate, method, t):
    print("all face test start")
    detect_error = 0
    identify_error = 0

    # 去除用于训练的那一张
    test_num = -1

    # 获取样本文件
    samples, known_face_encodings = _tool.get_known_name_encodings(filePath, rate)

    for sample in samples:
        # 获得图片路径
        image_path = filePath + '\\' + sample
        image_names = os.listdir(image_path)
        print(sample + 'starting identify')
        for image_name in image_names:
            image_test = face_recognition.load_image_file(image_path + '\\' + image_name)
            face_locations = face_recognition.face_locations(image_test)
            if len(face_locations) == 0:
                face_locations = face_recognition.face_locations(image_test, model="cnn")
                print("cpu run error, use GPU")
                detect_error += 1
            face_encoding_test = face_recognition.face_encodings(image_test, face_locations)[0]
            matches = _tool.compare_face(known_face_encodings, face_encoding_test, t, method)
            # print(matches)
            if True in matches:
                # index() 函数用于从列表中找出某个值第一个匹配项的索引位置
                first_match_index = matches.index(True)
                name = samples[first_match_index]
                print(sample + ' ' + image_name + ' recognition result: ' + name)
                if name != sample:
                    identify_error += 1
            if True not in matches:
                print("NO." + sample + '' + image_name + 'fails to identify')
                identify_error += 1
            test_num += 1
        print("NO." + sample + " is done")
    accuracy = 1 - (detect_error + identify_error) / test_num
    return {'test_num': test_num, 'detect_error': detect_error, 'identify_error': identify_error,
            'accuracy': accuracy}

# 输入数据为字典，增加识可信度
# def all_face_test(train_data, test_data)


# result = same_face_test(cas_Path, 'p', 0.984)
result = all_face_test(cas_Path, 0.7, 'm', 5.8)
