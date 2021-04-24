# -*- coding: utf-8 -*-
# _*_ coding:GBK _*_
# 摄像头头像识别
import base64

import pymongo
from flask import Flask, render_template, request, Response, jsonify
import tool

# from camera_pi import Camera

app = Flask(__name__)
# app.config['MONGODB_SETTINGS'] = {
#     'db': 'surveillance',
#     'host': 'localhost',
#     'port': 27017
# }
# db = MongoEngine(app)


# CORS(app, supports_credentials=True)
# 用户基本信息测
# @app.route('/userProfile', methods=["GET", "POST"])
# def get_profile():
#     if request.method =="GET":
#         name = request.args.get('name', '')
#         # 前端可用 userProfile?name = xxx获得相应信息
#         tool.log_record(None, "\n前端发送姓名请求（GET）")
#         userProfile = {'name': "Kaige", 'password': 123456}
#         return userProfile
#     elif request.method == "POST":
#         tool.log_record(None, "\n前端发送姓名请求（POST）")
#         name = request.json.get()
#         print(name)
#         return "收到POST请求2"

# 连接数据库
# 连接数据库
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["surveillance"]
collection = db["user_photo"]


# 主页路由
@app.route('/')  # 主页
def index():
    # jinja2模板，具体格式保存在index.html文件中
    tool.log_record(None, "\n主页已打开")
    return render_template('home.html')


# 新人脸登记
@app.route('/new_face/')
def new_face():
    tool.log_record(None, "\n人脸录入界面已打开")
    return render_template('face_login.html')


# # 前端向后端发送图片
# @app.route('/receiveImage/', methods=["POST"])
# def receive_image():
#     if request.method == "POST":
#         print("收到POST请求")
#         data = request.data.decode('utf-8')
#         json_data = json.loads(data)
#         name = json_data.get("name")
#         print("收到的姓名为" + name)
#         str_image = json_data.get("imgData")
#         img = base64.b64decode(str_image)
#         img_np = numpy.fromstring(img, dtype='uint8')
#         new_img_np = cv2.imdecode(img_np, 1)
#         cv2.imwrite('C:\\Users\\Kaige\\Desktop\\' + name + ".jpg", new_img_np)
#         cv2.imwrite('C:\\Users\\Kaige\\PycharmProjects\\pythonProject\\known_name' + name + ".jpg", new_img_np)
#         print(name + "的图片文件已写入")
#         tool.log_record(None, "\n" + name + "的图片已经存入")
#     return Response('upload')


# 后端获取图片
@app.route('/receiveImage/', methods=["POST"])
def receive_image():
    tool.deposit_image(collection)
    return Response('upload')


# 监视器路由
@app.route('/monitor/')
def monitor():
    print("网页监控界面已打开")
    tool.log_record(None, "\n网页监控界面已打开")
    return render_template('monitor.html')


# 监控视频接口
@app.route('/video_stream/')
def video_stream():
    tool.log_record(None, "\n收到前端视频流发送请求")
    return Response(tool.get_video(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
