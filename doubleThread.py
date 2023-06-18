from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QWidget
from PyQt5.QtCore import QThread, pyqtSignal, QMutex
from PyQt5.QtCore import Qt, QProcess
import dlib
import sys
import imagezmq
import redis
import cv2
import os
# import resource
import socket
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import socketio
from multiprocessing import Process, Manager
import simplejpeg
import traceback
import socket
import time
import tensorflow as tf
import random
import smtplib

# 创建背景减除器
backSub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=50, detectShadows=False)
# 调节背景减除器的参数
backSub.setHistory(100)  # 设置历史帧数为100帧
backSub.setVarThreshold(100)  # 设置前景掩码的阈值为30

jpeg_quality = 80  # 调整图片压缩质量，95%

detector = dlib.get_frontal_face_detector()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2) # 使用 validation_split 参数来指定验证集占总数据集的比例

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training') # 使用 subset 参数来指定训练集

validation_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation') # 使用 subset 参数来指定验证集
# 加载模型
face_recognition_model = tf.saved_model.load('D:/study/毕设/faceRecognition/mask_recognition-master(transed to be face_recognition)/face_recognition_saved_model')
# 获取模型输入和输出
model_input = face_recognition_model.signatures['serving_default'].inputs[0]
model_output = face_recognition_model.signatures['serving_default']

# 加载标签
class_indices = train_generator.class_indices
label_map = {v: k for k, v in class_indices.items()}

close_signal_global = pyqtSignal()

# # 接收发送端数据，输入发送端的ip地址
# image_hub = imagezmq.ImageHub(open_port='tcp://127.0.0.1:6000', REQ_REP=False)
class CameraThread(QThread):
    new_frame = pyqtSignal(QImage)
    def __init__(self):
        super(CameraThread, self).__init__()
        self.mutex = QMutex()
        self.running_face_mov = True
        # self.face_gain = False
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 10)  # 设置帧率为15fps
        # 192.168.100.104 为发送端主机ip地址
        self.sender = imagezmq.ImageSender(connect_to='tcp://127.0.0.1:6000', REQ_REP=False)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

    def run(self):
        cortime = []
        user_time = []
        while self.running_face_mov:
            ret, frame = self.cap.read()
            time_start = time.time()
            if not ret:
                break
            try:
                # 调整图像大小和预处理
                image = cv2.resize(frame, (224, 224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype('float32') / 255.0
                image = tf.expand_dims(image, axis=0)
                # 使用模型进行预测
                pred = model_output(tf.constant(image))['dense_17']
                # 获取预测结果
                # 加载标签
                class_names = list(class_indices.keys())  # 标签名字
                prediction = tf.argmax(pred, axis=1).numpy()[0]
                label = class_names[prediction]
                confidence = tf.nn.softmax(pred, axis=1).numpy()[0][prediction]
                # 在图像上显示结果
                cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                time_end = time.time()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 背景减除，得到前景掩模
                fgMask = backSub.apply(frame)
                # 形态学操作
                kernel = np.ones((5, 5), np.uint8)
                fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
                # 轮廓检测
                contours, hierarchy = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # 绘制检测到的轮廓
                for c in contours:
                    if cv2.contourArea(c) > 2000:
                        (x, y, w, h) = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 绘制人脸检测框
                faces = detector(gray, 1)
                for i, d in enumerate(faces):
                    cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 255), 3)

                jpg_buffer = simplejpeg.encode_jpeg(frame, quality=jpeg_quality,
                                                    colorspace='BGR')
                self.sender.send_jpg('frame', jpg_buffer)
                # 转换颜色空间
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                h, w, ch = frame.shape
                qimage = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
                self.mutex.lock()
                self.new_frame.emit(qimage)
                self.mutex.unlock()
            except Exception as e:
                print("Error in sending frame:", e)

    def stop(self):
        # self.face_gain = False
        self.running_face_mov = False
        self.cap.release()
        self.sender.close()

class Face_Col(QThread):
    f_c = pyqtSignal(QImage)
    close_signal = pyqtSignal()
    def __init__(self):
        super(Face_Col, self).__init__()
        self.mutex = QMutex()
        self.cap = cv2.VideoCapture(0)
        # self.cap.set(cv2.CAP_PROP_FPS, 10)  # 设置帧率为15fps
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

    def run(self):
        font = cv2.FONT_HERSHEY_SIMPLEX
        user_id = 0
        for user_key in self.redis_client.keys('user:*'):
            user_id += 1
        user_name = self.redis_client.hget(f'user:{user_id}', 'username').decode()
        os.mkdir('D:/study/毕设/code/train/' + str(user_name))
        os.mkdir('D:/study/毕设/code/valid/' + str(user_name))
        count = 0
        while (True):
            ret, frame = self.cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detector = dlib.get_frontal_face_detector()
            faces = detector(gray)
            if count < 60:
                cv2.putText(frame, "please shake your head slowly", (25, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "up and down", (150, 90), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            elif count > 59 and count < 110:
                cv2.putText(frame, "please shake your head slowly", (25, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "side to side", (150, 90), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                break
            if len(faces) > 0:
                for i, face in enumerate(faces):
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face_img = frame[y:y + h, x:x + w]
                    # 保存录入的图片
                    if count > 49 and count < 60:
                        cv2.imwrite('valid/' + str(user_name) + '/user.' + str(user_id) + '.' + str(count) + '.jpg',
                                    face_img)
                    elif count > 99 and count < 110:
                        cv2.imwrite('valid/' + str(user_name) + '/user.' + str(user_id) + '.' + str(count) + '.jpg',
                                    face_img)
                    else:
                        cv2.imwrite('train/' + str(user_name) + '/user.' + str(user_id) + '.' + str(count) + '.jpg',
                                    face_img)
                    count += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w, ch = frame.shape
            qimage = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888)
            self.mutex.lock()
            self.f_c.emit(qimage)
            self.mutex.unlock()
            cv2.waitKey(1)
        self.cap.release()
        self.mutex.lock()
        self.close_signal.emit()
        self.mutex.unlock()

#登录界面
class Ui_MainWindow(object):
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.camera_thread = CameraThread()
        self.camera_thread.new_frame.connect(self.update_image)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(843, 659)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(590, 210, 151, 41))
        self.lineEdit.setObjectName("lineEdit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(531, 220, 71, 20))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(540, 300, 81, 41))
        self.label_2.setObjectName("label_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(590, 300, 151, 41))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.lineEdit_2.setEchoMode(QtWidgets.QLineEdit.Password)  # 设置为密码形式
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(570, 400, 101, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(lambda :self.login_check(self.camera_thread))
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(570, 480, 101, 41))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(lambda :self.regist_page(self.camera_thread))
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(70, 140, 381, 381))
        self.label_3.setObjectName("label_3")
        self.label_3.setScaledContents(True)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.camera_thread.start()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "智能室内监控系统"))
        self.label.setText(_translate("MainWindow", "用户名"))
        self.label_2.setText(_translate("MainWindow", "密码"))
        self.pushButton.setText(_translate("MainWindow", "登录"))
        self.pushButton_2.setText(_translate("MainWindow", "注册"))
        self.label_3.setText(_translate("MainWindow", "waiting for the camera..."))

    def update_image(self, image):
        self.label_3.setPixmap(QPixmap.fromImage(image))

    def regist_page(self, camera_thread):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_Register(camera_thread)
        self.ui.setupUi(self.window)
        self.window.show()
        # self.ui.close_signal.connect(self.restart)

    def restart(self):
        self.window.close()
        self.camera_thread.start()

    def login_check(self, camera_thread):
        username = self.lineEdit.text()
        password = self.lineEdit_2.text()

        # 检查用户名和密码是否正确
        # 获取用户数据
        flag = True
        for user_key in self.redis_client.keys('user:*'):
            user_id = user_key[5:].decode()
            if username == self.redis_client.hget(f'user:{user_id}', 'username').decode():
                user_data = self.redis_client.hgetall(f'user:{user_id}')
                flag = False
                break
        if flag or password != user_data[b'password'].decode():
            # 如果用户名或密码不正确，则弹出消息框
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("用户名或密码不正确")
            msg.setWindowTitle("error")
            msg.exec_()
        else:
            self.login(camera_thread)

    def login(self, camera_thread):
        #self.close()
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_Monitor(camera_thread)
        self.ui.setupUi(self.window)
        self.window.show()

#注册界面
class Ui_Register(object):
    def __init__(self, camera_thread):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.camera_thread = camera_thread
        self.camera_thread.stop()
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(280, 110, 221, 41))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(280, 200, 221, 41))
        self.textEdit_2.setObjectName("textEdit_2")
        # self.textEdit_2.setEchoMode(QtWidgets.QTextEdit.Password)  # 设置为密码形式
        self.textEdit_3 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_3.setGeometry(QtCore.QRect(280, 360, 221, 41))
        self.textEdit_3.setObjectName("textEdit_3")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(280, 280, 131, 41))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(340, 460, 101, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(lambda :self.face_collect_check(self.camera_thread))
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(530, 367, 91, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.send_code)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(180, 120, 91, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(180, 210, 91, 31))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(180, 280, 91, 31))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(120, 360, 151, 31))
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.code = ''

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "智能室内监控系统"))
        self.comboBox.setItemText(0, _translate("MainWindow", "普通用户"))
        self.comboBox.setItemText(1, _translate("MainWindow", "管理员"))
        self.pushButton.setText(_translate("MainWindow", "下一步"))
        self.label.setText(_translate("MainWindow", "用户名"))
        self.label_2.setText(_translate("MainWindow", "密码"))
        self.label_3.setText(_translate("MainWindow", "用户类型"))
        self.label_4.setText(_translate("MainWindow", "验证码（管理员必填）"))
        self.pushButton_2.setText(_translate("MainWindow", "发送验证码"))

    def face_collect_check(self, camera_thread):
        username = self.textEdit.toPlainText()
        password = self.textEdit_2.toPlainText()
        code = self.textEdit_3.toPlainText()
        name_flag = True
        for user_key in self.redis_client.keys('user:*'):
            user_id = user_key[5:].decode()
            if username == self.redis_client.hget(f'user:{user_id}', 'username').decode():
                name_flag = False
                break
        if username == '':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("用户名不能为空！")
            msg.setWindowTitle("error")
            msg.exec_()
        elif password == '':
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("密码不能为空！")
            msg.setWindowTitle("error")
            msg.exec_()
        elif self.comboBox.currentText() == "管理员" and (code == '' or code != self.code):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("验证码错误！")
            msg.setWindowTitle("error")
            msg.exec_()
        elif not name_flag:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("用户名已存在，请更换！")
            msg.setWindowTitle("error")
            msg.exec_()
        else:
            if self.comboBox.currentText() == "管理员":
                user_type = 'admin'
            else:
                user_type = 'regular'
            k = 0
            for user_key in self.redis_client.keys('user:*'):
                k += 1
            user_id = k + 1
            # 将用户数据保存到Redis中
            self.redis_client.hmset(f'user:{user_id}',
                                        {'username': username, 'password': password, 'user_type': user_type})
            self.face_collect(camera_thread)

    def face_collect(self, camera_thread):
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_FaceCollect(camera_thread)
        self.ui.setupUi(self.window)
        self.window.show()

    def close_win(self):
        # self.close_signal.emit()
        self.window.close()

    def send_code(self):
        pass
        # randomlength=4
        # s=''
        # base_str = '0123456789'
        # length = len(base_str) - 1
        # for i in range(randomlength):
        #    s += base_str[random.randint(0, length)]
        # self.code = s
        # mail_host = "smtp.qq.com"
        # mail_sender = "3273675475@qq.com"  # 发件人邮箱
        # mail_license = "xftndswlywzxdbaj"  # 邮箱授权码
        # mail_receivers = self.redis_client.hget(f'user:{str(1)}', 'email').decode()  # 收件人邮箱，可以为多个收件人
        # mm = MIMEMultipart('related')
        # subject_content = """验证码"""  # 邮件主题
        # mm["From"] = "sender_name<3273675475@qq.com>"  # 设置发送者,注意严格遵守格式,里面邮箱为发件人邮箱
        # mm["To"] = "receiver_1_name<niuhkyuanshi@foxmail.com>"  # 设置接受者,注意严格遵守格式,里面邮箱为接受者邮箱
        # mm["Subject"] = Header(subject_content, 'utf-8')  # 设置邮件主题
        # body_content = """您好，有用户尝试创建管理员用户，现发送验证码如下："""  # 邮件正文内容
        # message_text = MIMEText(body_content + s, "plain", "utf-8")  # 构造文本,参数1：正文内容，参数2：文本格式，参数3：编码方式
        # mm.attach(message_text)  # 向MIMEMultipart对象中添加文本对象
        #
        # stp = smtplib.SMTP()  # 创建SMTP对象
        # stp.connect(mail_host, 25)  # 设置发件人邮箱的域名和端口，端口地址为25
        # stp.set_debuglevel(1)  # set_debuglevel(1)可以打印出和SMTP服务器交互的所有信息
        # stp.login(mail_sender, mail_license)  # 登录邮箱，传递参数1：邮箱地址，参数2：邮箱授权码
        # stp.sendmail(mail_sender, mail_receivers, mm.as_string())  # 发送邮件，传递参数1：发件人邮箱地址，参数2：收件人邮箱地址，参数3：把邮件内容格式改为str
        # stp.quit()  # 关闭SMTP对象
#系统主界面
class Ui_Monitor(object):
    def __init__(self, camera_thread):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.camera_thread = camera_thread
        self.camera_thread.new_frame.connect(self.update_image)
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(60, 40, 91, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(60, 80, 91, 31))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(120, 150, 551, 361))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(120, 50, 72, 15))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(120, 90, 72, 15))
        self.label_5.setObjectName("label_5")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "智能室内监控系统"))
        self.label.setText(_translate("MainWindow", "温度："))
        self.label_2.setText(_translate("MainWindow", "湿度："))
        self.label_3.setText(_translate("MainWindow", "waiting for the camera"))
        self.label_4.setText(_translate("MainWindow", "25℃"))
        self.label_5.setText(_translate("MainWindow", "36%"))

    def update_image(self, image):
        scaled_img = image.scaled(551, 361, Qt.KeepAspectRatio)
        self.label_3.setPixmap(QPixmap.fromImage(scaled_img))
#人脸数据收集
class Ui_FaceCollect(object):
    close_signal = pyqtSignal()
    def __init__(self, camera_thread):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.camera_thread = camera_thread

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(100, 60, 591, 431))
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(340, 510, 111, 41))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.collect_start)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.Face_Col = Face_Col()
        # self.Face_Col.start()
        self.Face_Col.close_signal.connect(self.close_cam)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "智能室内监控系统"))
        self.label.setText(_translate("MainWindow", "waiting for starting"))
        self.pushButton.setText(_translate("MainWindow", "开始"))

    def collect_start(self):
        self.Face_Col.start()
        self.Face_Col.f_c.connect(self.camera_open)

    def camera_open(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def close_cam(self):
        self.Face_Col.quit()
        self.Face_Col.wait()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("请返回登录界面重新登录！")
        msg.setWindowTitle("success")
        msg.exec_()
        # 停止相应的线程或子进程
        self.camera_thread.quit()
        self.camera_thread.wait()
        # # 释放资源
        # resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
        # # 重启相应的线程或子进程
        # QProcess.startDetached(sys.executable, sys.argv)
        python = sys.executable
        os.execl(python, python, *sys.argv)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(win)
    win.show()
    sys.exit(app.exec_())