# coding: utf-8
import face_recognition
import os
import random
import _tool


# 输入文件夹父目录，返回测试样本总数，HOG\CPU人脸检测错误数，HOG\CPU人脸检测错误率，人脸识别错误数，人脸检测错误率
def all_face_test(filePath, t):
    # 获取目录下所有文件夹的名称
    file_names = os.listdir(filePath)
    test_num = 0
    detect_error = 0
    indentify_error = 0
    known_face_names = []
    known_face_encodings = []
    t = 0.6
    print('start')
    # 遍历所有文件夹
    for file_name in file_names:
        # 获得图片路径
        image_path = filePath + '\\' + file_name
        image_names = os.listdir(image_path)
        # 随机选择选择文件夹内一张图片作为训练数据
        image = face_recognition.load_image_file(
            image_path + '\\' + image_names[random.randint(0, len(image_names) - 1)])
        # image = face_recognition.load_image_file(
        #     image_path + '\\' + image_names[1])
        face_locations = face_recognition.face_locations(image)
        # 如果识别不到，调用GPU
        if len(face_locations) == 0:
            face_locations = face_recognition.face_locations(image, model="cnn")
            print("cpu run error, use GPU")
            detect_error += 1
        # 注意extend和append的区别
        known_face_names.append(file_name)
        known_face_encodings.extend(face_recognition.face_encodings(image, face_locations))
        print('%s is done' % file_name)

    print('all faces have registered')
    print(known_face_names)

    for file_name in file_names:
        # 获得图片路径
        image_path = filePath + '\\' + file_name
        image_names = os.listdir(image_path)
        print(file_name + 'starting identify')
        for image_name in image_names:
            image_test = face_recognition.load_image_file(image_path + '\\' + image_name)
            face_locations = face_recognition.face_locations(image_test)
            if len(face_locations) == 0:
                face_locations = face_recognition.face_locations(image_test, model="cnn")
                print("cpu run error, use GPU")
                detect_error += 1
            #
            face_encoding_test = face_recognition.face_encodings(image_test, face_locations)[0]
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding_test, tolerance=t)
            if True in matches:
                # index() 函数用于从列表中找出某个值第一个匹配项的索引位置
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                print(file_name + ' ' + image_name + ' recognition result: ' + name)
                if name != file_name:
                    indentify_error += 1
            if True not in matches:
                print('fail to indentify')
                indentify_error += 1
            test_num += 1
    detect_error_rate = f'{detect_error / test_num:.3%}'
    indentify_error_rate = f'{indentify_error / test_num:.3%}'
    return test_num, detect_error, detect_error_rate, indentify_error, indentify_error_rate


# face_recognition全部人脸识别结果,
Path = 'C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\att-database-of-faces'
print("same face test start")

test_num, detect_error, detect_error_rate, identify_error, identify_error_rate = all_face_test(Path, 0.6)
_tool.result_print(test_num, detect_error, detect_error_rate, identify_error, identify_error_rate)
