# -*- coding: utf-8 -*-
import face_recognition

# 初始化， 测试用
def initialization():
    global known_face_encodings
    global known_face_names
    global face_locations
    global face_encodings
    global N

    KaigeZhu_image = face_recognition.load_image_file("C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\known\\me.jpg")
    KaigeZhu_face_encoding = face_recognition.face_encodings(KaigeZhu_image)[0]

    known_face_encodings = [
        KaigeZhu_face_encoding,
    ]

    known_face_names = [
        "KaigeZhu",
    ]
    face_names = []
    N = 1
    return known_face_encodings, known_face_names, N, face_names