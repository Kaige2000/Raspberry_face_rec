# _*_ coding:GBK _*_
# coding: utf-8
from flask import Flask, session
from front.front import front as front_blueprint
from background.background import background as background_blueprint

# ����APP
app = Flask(__name__)

# ����secret key����ֹCSR����
app.config['SECRET_KEY'] = '6d7f8h329rfjf'

# ע����ͼ
app.register_blueprint(front_blueprint)
app.register_blueprint(background_blueprint)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)


# # ��ҳ·��
# @app.route('/')  # ��ҳ
# def index():
#     # jinja2ģ�壬�����ʽ������index.html�ļ���
#     tool.log_record(None, "\n��ҳ�Ѵ�")
#     return render_template('home.html')


# # �������Ǽ�
# @app.route('/new_face/')
# def new_face():
#     tool.log_record(None, "\n����¼������Ѵ�")
#     return render_template('face_login.html')
#
#
# # ��˻�ȡͼƬ
# @app.route('/receive_Image/', methods=["POST"])
# def receive_image(collection):
#     tool.deposit_image(collection)
#     print('����ͼƬ')
#     return Response('up')


# # �Ա�ͼƬ��Ϣ���Ƿ��ظ�
# @app.route('/compareImage/', methods=["POST"])
# def compareImage():
#     # print(tool.compare_face())
#     result = tool.compare_face()
#     return render_template('face_login.html', result_list=result)
#
#
# # ������·��
# @app.route('/monitor/')
# def monitor():
#     print("��ҳ��ؽ����Ѵ�")
#     tool.log_record(None, "\n��ҳ��ؽ����Ѵ�")
#     return render_template('monitor.html')
#
#
# # �����Ƶ�ӿ�
# @app.route('/video_stream/')
# def video_stream():
#     tool.log_record(None, "\n�յ�ǰ����Ƶ����������")
#     return Response(tool.get_video(), mimetype="multipart/x-mixed-replace; boundary=frame")


# app.config['MONGODB_SETTINGS'] = {
#     'db': 'surveillance',
#     'host': 'localhost',
#     'port': 27017
# }
# db = MongoEngine(app)

# CORS(app, supports_credentials=True)
# �û�������Ϣ��
# @app.route('/userProfile', methods=["GET", "POST"])
# def get_profile():
#     if request.method == "GET":
#         name = request.args.get('name', '')
#         # ǰ�˿��� userProfile?name = xxx�����Ӧ��Ϣ
#         tool.log_record(None, "\nǰ�˷�����������GET��")
#         userProfile = {'name': "Kaige", 'password': 123456}
#         return render_template('test.html', data=userProfile)
#     elif request.method == "POST":
#         tool.log_record(None, "\nǰ�˷�����������POST��")
#         # name = request.json.get()
#         # print(name)
#         return "�յ�POST����2"
