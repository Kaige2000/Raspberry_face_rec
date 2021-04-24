# 人脸鉴定

import face_recognition

#将jpg文件加载到numpy数组中
konwn_image = face_recognition.load_image_file("C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\known\\me.jpg")
#要识别的图片
unknown_image = face_recognition.load_image_file("C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\unknown\\us.jpg")

#获取每个图像文件中每个面部的面部编码
#由于每个图像中可能有多个面，所以返回一个编码列表。
konwn_face_encoding = face_recognition.face_encodings(konwn_image)[0]


print(konwn_face_encoding)

#print("unknown_face_encoding :{}".format(unknown_face_encoding))

#结果是True/false的数组，


