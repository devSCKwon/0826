from PyQt5.QtWidgets import QPushButton
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog
import cv2


import mediapipe
import numpy as np
import sys

from cvzone.HandTrackingModule import HandDetector
import csv
form_class = uic.loadUiType("./ui/gen.ui")[0]



class UIToolTab(QWidget,form_class):  ###########################화면구성
    def __init__(self, parent=None):
        super(UIToolTab, self).__init__(parent)
        self.setupUi(self)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        #self.setGeometry(0, 0, 661, 733)
        self.setFixedSize(661, 578)
        self.startUIToolTab()

    def startUIToolTab(self):  ####### 페이지 별 동작함수 구현
        self.ToolTab = UIToolTab(self)
        #self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowTitle("UIToolTab")
        self.setCentralWidget(self.ToolTab)

        self.flag = 0
        self.count=0
        self.frame=0
        self.ToolTab.save_button.clicked.connect(self.save_function)
        self.ToolTab.stop_button.clicked.connect(self.stop_function)
        self.capture = cv2.VideoCapture(0)


        self.detector = HandDetector(maxHands=1, detectionCon=0.75)
        self.show()

        self.start_webcam()
    def save_function(self):

        self.flag=1



    def stop_function(self):
        if self.flag==1:
            self.count =self.count+1
            self.frame=0
        self.flag = 0


    def start_webcam(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)



    def update_frame(self):

        ret, self.image = self.capture.read()
        self.image = cv2.resize(self.image, (320, 240))
        self.image = cv2.flip(self.image, 1)
        input_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        hands, self.image = self.detector.findHands(self.image, flipType=False)

        self.name = self.ToolTab.name.text()
        if hands:
            lm_list = hands[0]['lmList']
            lm_list = np.array(lm_list)
            v1 = lm_list[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
            v2 = lm_list[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
            v = v2 - v1  # (20,3) 팔목과 각 손가락 관절 사이의 벡터를 구한다.

            v = v / np.expand_dims(np.linalg.norm(v, axis=1), axis=-1)  # 유닛벡터 구하기 벡터/벡터의 길이

            angle = np.arccos(np.einsum('nt,nt->n',
                                        v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                        :]))  # [15,] 유닛벡터를 내적한 값의 아크 코사인을 구하면 각도를 구할 수 있다.
            angle = np.degrees(angle)  # Convert radian to degree
            angle = np.append(angle, np.array(self.count))

            print(angle)
            if self.flag == 1:
                framecount = "count: {}".format(self.frame)
                self.ToolTab.result.setText(framecount)
                self.frame = self.frame + 1
                str = self.name + '.csv'
                f = open(str, 'a', encoding='utf-8', newline='')
                wr = csv.writer(f)
                wr.writerow(angle)
                f.close()

        self.displayImage(self.image, 1)

    def stop_webcam(self):
        self.timer.stop()
        # self.capture.release()

    def displayImage(self, img, window=1):
        qformat = QImage.Format_Indexed8
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window == 1:
            self.ToolTab.imglabel.setPixmap(QPixmap.fromImage(outImage))
            self.ToolTab.imglabel.setScaledContents(True)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    sys.exit(app.exec_())