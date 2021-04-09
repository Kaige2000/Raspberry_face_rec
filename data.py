# -*- coding: utf-8 -*-
import face_recognition

KaigeZhu_image = face_recognition.load_image_file("C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\known\\me.jpg")
KaigeZhu_face_encoding = face_recognition.face_encodings(KaigeZhu_image)[0]

known_face_encodings = [
    KaigeZhu_face_encoding,
]

# 人物名称的集合
known_face_names = [
    "KaigeZhu",
]