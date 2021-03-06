# -*- coding: utf-8 -*-
#  识别图片中的所有人脸并显示出来

# 导入pil模块 ，可用命令安装 apt-get install python-Imaging
from PIL import Image
# 导入face_recogntion模块，可用命令安装 pip install face_recognition
import face_recognition

# 将jpg文件加载到numpy 数组中
image = face_recognition.load_image_file("C:\\Users\\Kaige\\OneDrive\\学习\\毕业设计\\FaceDetect\\CASIA-FaceV5\\009\\009_2.bmp")
# 使用默认的给予HOG模型查找图像中所有人脸
# 这个方法已经准确了，但还是不如CNN模型那么准确，因为没有使用GPU加速(树莓派没有单独的GPU，后续可以考虑使用因特尔神经棒)
# if len(face_recognition.face_locations(image)) == 0:
#     print('CPU wrong;using GPU')
#     face_locations = face_recognition.face_locations(image, model="cnn")
# 使用CNN模型
face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
print(face_locations)
# # 打印：我从图片中找到了 多少 张人脸
# print("{} face(s) can be found in the picture.".format(len(face_locations)))
# print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
# # 循环找到的所有人脸

for face_location in face_locations:
    # 打印每张脸的位置信息
    top, right, bottom, left = face_location
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()
