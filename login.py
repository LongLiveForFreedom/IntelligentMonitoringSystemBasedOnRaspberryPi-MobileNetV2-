from flask import Flask, render_template, request, redirect, url_for, session, abort, Response
from functools import wraps
import redis
import cv2
import numpy as np
import simplejpeg
import imagezmq
import traceback
import time

r = redis.Redis(host = '127.0.0.1', port = '6379')

app = Flask(__name__, template_folder='templates')
app.secret_key = 'secret-key'



# 装饰器：限制只有管理员用户可以访问
def admin_required(func):
    @wraps(func)
    def decorated_view(*args, **kwargs):
        user_id = session.get('user_id')
        if not user_id:
            return redirect(url_for('login'))

        # 从Redis中获取用户类型
        user_type = r.hget(f'user:{user_id}', 'user_type')
        if user_type != b'admin':
            abort(403)

        return func(*args, **kwargs)
    return decorated_view

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # 获取用户数据
        flag = True
        for user_key in r.keys('user:*'):
            user_id = user_key[5:].decode()
            if username == r.hget(f'user:{user_id}', 'username').decode():
                user_data = r.hgetall(f'user:{user_id}')
                flag = False
                break
        if flag:
            return 'Invalid username or password'
        elif password != user_data[b'password'].decode():
            return 'Invalid username or password'

        # 将用户ID存储在session中
        #session['user_id'] = user_data[b'username']
        session['user_id'] = user_id

        return redirect(url_for('index'))

    return render_template('login.html')

# 用户退出登录
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))

# 用户主页
@app.route('/')
def index():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('login'))

    # 获取当前用户数据
    #user_data = r.hgetall(f'user:{user_id}')
    user_data = {
        'user_id': user_id,
        'username': r.hget(f'user:{user_id}', 'username').decode(),
        'user_type': r.hget(f'user:{user_id}', 'user_type').decode(),
        'email': r.hget(f'user:{user_id}', 'email').decode()
    }

    return render_template('user.html', user_data=user_data)


# 管理员用户列表
@app.route('/admin/users')
@admin_required
def admin_users():
    users = []
    for user_key in r.keys('user:*'):
        #user_data = r.hgetall(user_key)
        user_id = user_key[5:].decode()
        user_data = {
            'user_id': user_id,
            'username': r.hget(f'user:{user_id}', 'username').decode(),
            'user_type': r.hget(f'user:{user_id}', 'user_type').decode()
        }
        users.append(user_data)

    return render_template('admin_users.html', user_data=users)

# 用户增加
@app.route('/admin/users/add/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def admin_users_add(user_id):
    user_data = {
        'user_id': user_id,
        'username': r.hget(f'user:{user_id}', 'username').decode(),
        'user_type': r.hget(f'user:{user_id}', 'user_type').decode()
    }
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_type = request.form['user_type']

        # 生成新的用户ID
        k = 0
        for user_key in r.keys('user:*'):
            k += 1
        user_id = k + 1

        # 将用户数据保存到Redis中
        r.hmset(f'user:{user_id}', {'username': username, 'password': password, 'user_type': user_type})

        return redirect(url_for('admin_users'))

    return render_template('admin_users_add.html', user_data=user_data)

# 管理员用户编辑
@app.route('/admin/users/edit/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def admin_users_edit(user_id):
    #user_data = r.hgetall(f'user:{user_id}')
    user_data = {
        'user_id': user_id,
        'username': r.hget(f'user:{user_id}', 'username').decode(),
        'user_type': r.hget(f'user:{user_id}', 'user_type').decode()
    }
    if not user_data:
        abort(404)

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_type = request.form['user_type']

        # 更新用户数据
        r.hmset(f'user:{user_id}', {'username': username ,'password': password, 'user_type': user_type})

        return redirect(url_for('admin_users'))

    return render_template('admin_users_edit.html', user_data=user_data)

# 管理员用户删除
@app.route('/admin/users/delete/<int:user_id>', methods=['GET', 'POST'])
@admin_required
def admin_users_delete(user_id):
    user_data = {
        'user_id': user_id,
        'username': r.hget(f'user:{user_id}', 'username').decode(),
        'user_type': r.hget(f'user:{user_id}', 'user_type').decode()
    }
    if request.method == 'POST':
        r.delete(f'user:{user_id}')
        return redirect(url_for('admin_users'))

    return render_template('admin_users_delete.html', user_id=user_id, user_data=user_data)

#实时监控画面
@app.route('/realtime')
def realtime():
    # 视图函数返回模板，模板会在浏览器中渲染
    return render_template('realtime.html')

def gen_frames():
    # 接收发送端数据，输入发送端的ip地址
    image_hub = imagezmq.ImageHub(open_port='tcp://172.20.10.14:6000', REQ_REP=False)
    frame_count = 1
    time1 = 0
    while True:
        try:
            time1 = time.time() if frame_count == 1 else time1
            name, image = image_hub.recv_jpg()

            # Decode the JPEG image into a BGR image using simplejpeg
            image = simplejpeg.decode_jpeg(image, colorspace='BGR')

            # Convert the BGR image to a JPEG binary string
            ret, buffer = cv2.imencode('.jpg', image)
            jpeg_image = buffer.tobytes()

            # Yield the JPEG image frame as a multipart HTTP response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_image + b'\r\n')

        except:
            print(traceback.format_exc())
            break

@app.route('/video_feed')
def video_feed():
    # 视图函数返回响应，响应中包含视频流数据
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')