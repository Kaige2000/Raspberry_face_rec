import time
import cv2
import data
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
import face_recognition
import numpy as np


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


# def get_video():
#     video_capture = cv2.VideoCapture(1)
#     log_record(None, "\n后端摄像头已经打开")
#     flag = 1
#     while True:
#         ret, frame = video_capture.read()
#         # frame = video_face_rec(frame)
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         frame = jpeg.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         if flag == 1:
#             log_record(None, "\n视频帧已经发送")
#         flag = 0
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     video_capture.release()
#     cv2.destroyAllWindows()

def get_video():
    video_capture = cv2.VideoCapture(1)
    log_record(None, "\n后端摄像头已经打开")

    # 本地图片
    KaigeZhu_image = face_recognition.load_image_file("C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\known\\me.jpg")
    KaigeZhu_face_encoding = face_recognition.face_encodings(KaigeZhu_image)[0]
    log_record(None, "\n图片已经编码")
    known_face_encodings = [
        KaigeZhu_face_encoding,
    ]

    known_face_names = [
        "KaigeZhu",
    ]

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    log_record(None, "\n准备识别")
    flag = 1
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if flag == 1:
            log_record(None, "\n预处理完成")
        if process_this_frame:
            # 根据encoding来判断是不是同一个人，是就输出true，不是为flase
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # 默认为unknown
                # compare对比待识别向量与已知向量的欧拉距离
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]
                face_names.append(name)
        if flag == 1:
            log_record(None, "\n处理完一帧")

        process_this_frame = not process_this_frame
        # 将捕捉到的人脸显示出来
        if flag == 1:
            log_record(None, "\n准备绘图")

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
        if flag == 1:
            log_record(None, "\n已经贴上标签")

        ret, jpeg = cv2.imencode('.jpg', frame)
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


get_video()

# def video_face_rec(frame):
#     face_locations = []
#     face_encodings = []
#     face_names = []
#     process_this_frame = True
#     frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#     rgb_small_frame = frame[:, :, ::-1]
#     # Only process every other frame of video to save time
#
#     if process_this_frame:
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
#
#         face_names = []
#         for face_encoding in face_encodings:
#             # 默认为unknown
#             # compare对比待识别向量与已知向量的欧拉距离
#             matches = face_recognition.compare_faces(data.known_face_encodings, face_encoding, tolerance=0.6)
#             name = "Unknown"
#             if True in matches:
#                 first_match_index = matches.index(True)
#                 name = data.known_face_names[first_match_index]
#             face_names.append(name)
#
#     # 将捕捉到的人脸显示出来
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         # Scale back up face locations since the frame we detected in was scaled to 1/4 size
#         top *= 4
#         right *= 4
#         bottom *= 4
#         left *= 4
#
#         # 矩形框
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         cv2.rectangle(frame, (right, top), (right, top), (0, 0, 0), 5, cv2.FILLED)
#
#         #加上标签
#         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 5, bottom - 6), font, 1.0, (255, 255, 255), 1)
#     return frame
