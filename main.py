
# coding: utf-8
from flask import Flask, session
from front.front import front as front_blueprint
from background.background import background as background_blueprint

# 创建AP
app = Flask(__name__)

# 生成secret key，防止CSR攻击
app.config['SECRET_KEY'] = '6d7f8h329rfjf'

# 注册蓝图
app.register_blueprint(front_blueprint)
app.register_blueprint(background_blueprint)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
    # app.run(host="0.0.0.0", debug=False, threaded=False, processes=5)



