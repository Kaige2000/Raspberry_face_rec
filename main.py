# _*_ coding:GBK _*_
# coding: utf-8
from flask import Flask, session
from front.front import front as front_blueprint
from background.background import background as background_blueprint

# 创建APP
app = Flask(__name__)

# 生成secret key，防止CSR攻击
app.config['SECRET_KEY'] = '6d7f8h329rfjf'

# 注册蓝图
app.register_blueprint(front_blueprint)
app.register_blueprint(background_blueprint)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)


# # 主页路由
# @app.route('/')  # 主页
# def index():
#     # jinja2模板，具体格式保存在index.html文件中
#     tool.log_record(None, "\n主页已打开")
#     return render_template('home.html')


# # 新人脸登记
# @app.route('/new_face/')
# def new_face():
#     tool.log_record(None, "\n人脸录入界面已打开")
#     return render_template('face_login.html')
#
#
# # 后端获取图片
# @app.route('/receive_Image/', methods=["POST"])
# def receive_image(collection):
#     tool.deposit_image(collection)
#     print('处理图片')
#     return Response('up')


# # 对比图片信息，是否重复
# @app.route('/compareImage/', methods=["POST"])
# def compareImage():
#     # print(tool.compare_face())
#     result = tool.compare_face()
#     return render_template('face_login.html', result_list=result)
#
#
# # 监视器路由
# @app.route('/monitor/')
# def monitor():
#     print("网页监控界面已打开")
#     tool.log_record(None, "\n网页监控界面已打开")
#     return render_template('monitor.html')
#
#
# # 监控视频接口
# @app.route('/video_stream/')
# def video_stream():
#     tool.log_record(None, "\n收到前端视频流发送请求")
#     return Response(tool.get_video(), mimetype="multipart/x-mixed-replace; boundary=frame")


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
#     if request.method == "GET":
#         name = request.args.get('name', '')
#         # 前端可用 userProfile?name = xxx获得相应信息
#         tool.log_record(None, "\n前端发送姓名请求（GET）")
#         userProfile = {'name': "Kaige", 'password': 123456}
#         return render_template('test.html', data=userProfile)
#     elif request.method == "POST":
#         tool.log_record(None, "\n前端发送姓名请求（POST）")
#         # name = request.json.get()
#         # print(name)
#         return "收到POST请求2"
