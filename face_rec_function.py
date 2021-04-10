# -*- coding: utf-8 -*-
import face_recognition
import cv2
from tool import log_record


def initialization():
    # 本地图片
    global known_face_encodings
    global known_face_names
    global face_locations
    global face_encodings
    global face_names
    global N

    KaigeZhu_image = face_recognition.load_image_file("C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\known\\me.jpg")
    KaigeZhu_face_encoding = face_recognition.face_encodings(KaigeZhu_image)[0]

    known_face_encodings = [
        KaigeZhu_face_encoding,
    ]

    known_face_names = [
        "KaigeZhu",
    ]
    face_locations = []
    face_encodings = []
    face_names = []
    N = 1


def compare_label(frame, N, face_locations, face_encodings, face_names):
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]
    # 根据encoding来判断是不是同一个人，是就输出true，不是为flase
    face_locations = face_recognition.face_locations(rgb_small_frame)

    if N == 1:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        for face_encoding in face_encodings:
            # 默认为unknown
            # compare对比待识别向量与已知向量的欧拉距离
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            face_names.append(name)
    if N < 10:
        N = N +1
    if N == 10:
        N = 1
    print(N)

    # 将捕捉到的人脸显示出来
    # zip用于打包元组
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        # 矩形框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (right, top), (right, top), (0, 0, 0), 5, cv2.FILLED)
        # 加上标签
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 5, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return frame, N




initialization()
video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    (frame, N) = compare_label(frame, N, face_locations, face_encodings,face_names)
    cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
