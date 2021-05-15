# -*- coding: utf-8 -*-
# import pymongo
from flask import Blueprint, render_template, Response, session, redirect, url_for, flash
from . import tool


# 设定蓝图，并指定模板目录
front = Blueprint('front', __name__, template_folder='templates')

# # 连接数据库测试
# client = pymongo.MongoClient("mongodb://localhost:27017/")
# db = client["surveillance"]
# collection = db["user_photo"]


# 主页路由, 引用函数名
@front.route('/home/')  # 主页
def home():
    if not session.get('user'):
        flash('您还没有登录', 'danger')
        tool.log_record(None, "\n stranger tries to visit")
        return redirect(url_for('background.login'))
    tool.log_record(None, "\n home page is on")
    return render_template('home.html')


# 新人脸登记
@front.route('/new_face/')
def new_face():
    if not session.get('user'):
        flash('您还没有登录', 'danger')
        tool.log_record(None, "\n stranger tries to visit")
        return redirect(url_for('background.login'))
    tool.log_record(None, "\n face login page is on")
    return render_template('face_login.html')


# 后端获取图片
@front.route('/receive_Image/', methods=["POST"])
def receive_image():
    tool.deposit_image()
    # print('deal picture')
    return Response('up')


# 对比图片信息，是否重复
@front.route('/compareImage/', methods=["POST"])
def compareImage():
    # print(tool.compare_face())
    result = tool.compare_face()
    return render_template('face_login.html', result_list=result)


# 监视器路由
@front.route('/monitor/')
def monitor():
    if not session.get('user'):
        flash('您还没有登录', 'danger')
        return redirect(url_for('background.login'))
    print("monitor is on")
    tool.log_record(None, "\nmonitor is on")
    return render_template('monitor.html')


# 监控视频接口
@front.route('/video_stream/')
def video_stream():
    tool.log_record(None, "\nget video request")
    return Response(tool.get_video(), mimetype="multipart/x-mixed-replace; boundary=frame")
