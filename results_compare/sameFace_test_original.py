# coding: utf-8
import face_recognition
import os
import random


# Path = 'C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\att-database-of-faces'
# 输入文件夹父目录，返回测试样本总数，HOG\CPU人脸检测错误数，HOG\CPU人脸检测错误率，人脸识别错误数，人脸检测错误率
def same_face_test(filePath, t):
    # 获取目录下所有文件夹的名称
    file_names = os.listdir(filePath)
    test_num = 0
    dectect_error = 0
    indentify_error = 0

    # 遍历所有文件夹
    for file_name in file_names:
        # 获得图片路径
        image_path = filePath + '\\' + file_name
        image_names = os.listdir(image_path)

        # 随机选择选择文件夹内一张图片作为训练数据
        # image_train = str(random.randint(1, 9))
        # 将jpg文件加载到numpy 数组中

        image = face_recognition.load_image_file(
            image_path + '\\' + image_names[random.randint(0, len(image_names) - 1)])
        face_locations = face_recognition.face_locations(image)
        # 如果识别不到，调用GPU
        if len(face_locations) == 0:
            face_locations = face_recognition.face_locations(image, model="cnn")
            print("cpu run error, use GPU")
            dectect_error += 1

        face_encoding_train = face_recognition.face_encodings(image, face_locations)

        # print(face_encoding_train)
        for image_name in image_names:
            image_test = face_recognition.load_image_file(image_path + '\\' + image_name)
            face_locations = face_recognition.face_locations(image_test)

            if len(face_locations) == 0:
                face_locations = face_recognition.face_locations(image_test, model="cnn")
                print("cpu run error, use GPU")
                dectect_error += 1

            face_encoding_test = face_recognition.face_encodings(image_test, face_locations)[0]
            result = face_recognition.compare_faces(face_encoding_train, face_encoding_test, tolerance=t)
            # print(result)
            if not result[0]:
                indentify_error += 1
            print(" Name " + file_name + " NO. " + image_name + ' result: ' + str(result))
            test_num += 1
        print("NO." + file_name + " is done")

    dectect_error_rate = f'{dectect_error / test_num:.3%}'
    indentify_error_rate = f'{indentify_error / test_num:.3%}'

    return test_num, dectect_error, dectect_error_rate, indentify_error, indentify_error_rate


Path = 'C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\att-database-of-faces'
print("same face test start")
test_num, detect_error, detect_error_rate, identify_error, identify_error_rate = same_face_test(Path, 0.6)
print('the num of sample is %d' % test_num)
print('detect error num is %d, detect error rate is %s' % (detect_error, detect_error_rate))
print('identify error num is %d, identify error rate is %s' % (identify_error, identify_error_rate))


