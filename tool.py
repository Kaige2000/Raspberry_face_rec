import base64
import time
import cv2
import numpy as np
import data
import face_recognition
from flask import request
import json


# import os
# # coding=utf-8
# # with open("log.txt", "w") as f:
# #     f.write("")
# # dirs = ''
# # if not os.path.exists(dirs):
# #     os.makedirs(dirs)
# #
# # filename = ''
# # if not os.path.exists(filename):
# #     os.system(r"touch {}".format(path))#调用系统命令行来创建文件

# 获取操作时间

def get_time(localtime=None):
    localtime = time.asctime(time.localtime(time.time()))
    return localtime


# 文本写入，测试用
def log_record(path=None, record=None):
    if path == None:
        path = "log.txt"
    try:
        f = open(path, 'r')
        f.close()
    except IOError:
        f = open(path, 'w')
    record_time = get_time()
    f = open(path, "a")
    f.write(record + "\000操作时间:\000" + record_time + "\n")
    f.close()


# 识别人脸并标记
def compare_label(frame, N, known_face_encodings, known_face_names, face_names):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    # 根据encoding来判断是不是同一个人，是就输出true，不是为flase
    face_locations = face_recognition.face_locations(rgb_small_frame)
    # 每十帧检测一帧
    if N == 1:
        face_names = []
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        for face_encoding in face_encodings:
            '''
            compare对比待识别向量与已知向量的欧拉距离
            
            向量的相似度判定
            1. 欧氏距离
            2. 曼哈顿距离
            3. 切比雪夫距离
            4. 闵可夫斯基距离
            5. 标准化欧氏距离
            6. 马氏距离
            7. 夹角余弦
            
            其他的分类器
            KNN分类器
            SVM分类器
            贝叶斯分类器
            '''
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
            # 标签默认为unknown
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)
    if N < 15:
        N = N + 1
    if N == 15:
        N = 1
    # print(N)
    # 将捕捉到的人脸显示出来 zip用于打包元组
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # 矩形框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (right, top), (right, top), (0, 0, 0), 5, cv2.FILLED)
        # 加上标签
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 5, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return frame, N, face_names


# 获取识别后的视频帧
def get_video():
    known_face_encodings, known_face_names, N, face_names = data.initialization()
    video_capture = cv2.VideoCapture(1)
    log_record(None, "\n准备识别")
    flag = 1
    while True:
        ret, frame = video_capture.read()
        (frame, N, face_names) = compare_label(frame, N, known_face_encodings, known_face_names, face_names)
        ret, jpeg = cv2.imencode('.jpg', frame)
        # cv2.imshow("capture", frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        if flag == 1:
            log_record(None, "\n视频帧已经发送")
        flag = 0
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


def deposit_image(collection_name):
    collection = collection_name
    if request.method == "POST":
        print("收到POST请求")
        data = request.data.decode('utf-8')
        json_data = json.loads(data)
        name = json_data.get("name")
        print("收到的姓名为" + name)
        str_image = json_data.get("imgData")
        img = base64.b64decode(str_image)
        img_np = np.fromstring(img, dtype='uint8')
        new_img_np = cv2.imdecode(img_np, 1)
        # cv2.imwrite('C:\\Users\\Kaige\\Desktop\\' + name + ".jpg", new_img_np)
        save_address = 'C:\\Users\\Kaige\\PycharmProjects\\pythonProject\\known_name\\'
        cv2.imwrite(save_address + name + ".jpg", new_img_np)
        log_record(None, "\n" + name + "的图片已经存入服务器本地")
        picture_address = {"name": name, "photo_address": save_address}
        collection.insert_one(picture_address)
        log_record(None, "\n" + name + "的图片地址已经存入数据库")



