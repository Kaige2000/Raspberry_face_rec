from flask import Blueprint, render_template, request, flash, redirect, url_for, session

background = Blueprint('background', __name__, template_folder='views')


@background.route('/login', methods=['GET', 'POST'])
def login():
    # �򿪽�����GET����
    print(request.method)
    if request.method == 'GET':
        return render_template('Login.html')

    # ������POST����
    elif request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'admin' and password == '111111':
            flash('login success', 'success')
            session['user'] = 'admin'
            return redirect(url_for('front.home'))

        else:
            # �ض�����ԭ����
            flash('fail to login', 'danger')
            return redirect(url_for('background.login'))

    return render_template('Login.html')
