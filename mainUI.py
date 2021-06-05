from PyQt5.QtWidgets import QApplication, QMainWindow, QCalendarWidget, \
    QGraphicsDropShadowEffect, QWidget

from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QFileDialog, QListWidgetItem
from PyQt5.QtGui import QPixmap, QImage, QBrush, QColor, QIcon
from PyQt5 import QtCore
import threading
from functools import wraps
import configparser
import datetime
# import copy
import time
import sys
import cv2
import numpy as np
import os
from filter import ImageHolder

from ui.main_ui import Ui_MainWindow


class ImageViewer:
    def __init__(self, window):
        self.window = window
        self.ui = window.ui
        self.image = None
        pass

    def window_holder(self):
        self.WIN_WIDTH = self.window.width()
        self.WIN_HEIGHT = self.window.height()
        x_left, y_up = 0, 0
        x_right = 183
        y_down = 0
        self.x_shift = self.WIN_WIDTH - x_right
        self.y_shift = self.WIN_HEIGHT - y_down
        # print(self.WIN_WIDTH, self.WIN_HEIGHT)

        # self.FRAME_SIZE = (x_left, y_up, self.x_shift, self.y_shift)
        self.FULL_FRAME_SIZE = (x_left + 2, y_up, self.WIN_WIDTH - x_right, self.WIN_HEIGHT - y_down)
        # print(self.FRAME_SIZE)
        # self.ui.image.setGeometry(QtCore.QRect(*self.FRAME_SIZE))
        self.image_update()

    def image_update(self):
        def image_processing():
            image = self.image.copy()
            self.ORIGIN_IMSIZE = image.shape[:2]
            self.K = min(self.y_shift / image.shape[0], self.x_shift / image.shape[1])
            # print(self.K)
            self.NEW_IMSIZE = (round(np.array(image).shape[0] * self.K), round(np.array(image).shape[1] * self.K))

            image = cv2.resize(image, (self.NEW_IMSIZE[1], self.NEW_IMSIZE[0]))
            # height, width, channel = image.shape
            bytesPerLine = 3 * image.shape[1]
            qImg = QImage(image.data, image.shape[1], image.shape[0], bytesPerLine, QImage.Format_BGR888)
            return qImg

        if self.image is None: return
        self.ui.label_image.setPixmap(QPixmap(image_processing()))


class OpenFileHolder:
    def __init__(self, window: QMainWindow):
        self.ui = window.ui
        self.window = window
        self.path = None
        self.type = None
        self.image_types = ['jpg',
                            'jpeg',
                            'png',
                            'bmp']
        self.image = None
        self.ui.pushButton_save.clicked.connect(self.save_image)
        self.ui.pushButton_open.clicked.connect(self.open_file)

    def open_file(self, param, fname=None):

        if fname is None:
            fname = QFileDialog.getOpenFileName(self.window, 'Открыть файл',
                                                f'{os.path.dirname(__file__)}/',
                                                "Изображение (*.jpg *.jpeg *.png *.bmp)")[0]
        if fname:
            self.path = fname
            self.type = fname.split('.')[-1]
            if self.type in self.image_types:
                self.window.ImH.load_img(self.path)
                self.image = self.window.ImH.src_img
                self.window.window_holder.image = self.image
                self.window.window_holder.image_update()
                return self.path, self.image

    def save_image(self):
        if self.window.ImH.dst_img is not None:
            fname = QFileDialog.getSaveFileName(self.window, 'Save File', f'{os.path.dirname(__file__)}/',)[0]
            if fname:
                if len(fname.split('.')) == 1:
                    fname += '.png'
                cv2.imwrite(fname, self.window.ImH.dst_img)
                return 0


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        central_widget = QWidget()
        central_widget.setLayout(self.ui.mainLayout)  # main layout
        self.setCentralWidget(central_widget)

        self.modules = []
        self.window_holder = ImageViewer(self)
        self.open_file_holder = OpenFileHolder(self)
        self.ImH = ImageHolder()

        # self._reset_()
        self.ui.pushButton_analyze.clicked.connect(self.aplly_filters)
        self.ui.pushButton_analyze_folder.clicked.connect(self.analyze_folder)

    def _reset_(self):
        self.window_holder = ImageViewer(self)
        self.open_file_holder = OpenFileHolder(self)

    def aplly_filters(self):
        if self.ImH.src_img is None:
            return
        d, img = self.ImH.apply_filters()
        print(d)
        self.ui.textEdit_lines.clear()
        self._set_add_info_(d)
        self.window_holder.image = img
        self.window_holder.image_update()

    def _set_add_info_(self, d):
        self.ui.textEdit_lines.setText(
            "№: x1, y1, x2, y2\n" + "".join(f"{i}: {line[2]}\n" for i, line in enumerate(d)))

    def analyze_folder(self):
        IMAGE_DIR = str(QFileDialog.getExistingDirectory(self, "Выбор дериктории"))
        if IMAGE_DIR:
            self.ui.textEdit_lines.setText("Ждите")
            self.repaint()
            for i, filename in enumerate(os.listdir(IMAGE_DIR)):
                d, img = self.ImH.apply_filters(IMAGE_DIR + '/' + filename)
                # self.window_holder.image = self.ImH.dst_img
                # self.window_holder.image_update()
                # self._set_add_info_(d)
                # self.repaint()
        self.ui.textEdit_lines.setText("Готово")

    def resizeEvent(self, *args, **kwargs):
        if self.window_holder:
            self.window_holder.window_holder()


if __name__ == '__main__':
    app = QApplication([])
    application = MainWindow()
    application.show()
    sys.exit(app.exec())
