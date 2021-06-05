from PyQt5.QtWidgets import QApplication, QMainWindow, QCalendarWidget, \
    QGraphicsDropShadowEffect, QWidget
from PyQt5 import QtCore
import threading
from functools import wraps
import configparser
import datetime
# import copy
import time
import sys
# import os

from ui.main_ui import Ui_MainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        central_widget = QWidget()
        central_widget.setLayout(self.ui.mainLayout)  # main layout
        self.setCentralWidget(central_widget)

        self.setLayout(self.ui.mainLayout)
        self.main_modules = None
        self.list_threads = []


if __name__ == '__main__':
    app = QApplication([])
    application = MainWindow()
    application.show()
    sys.exit(app.exec())
