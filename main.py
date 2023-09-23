from PyQt5 import QtWidgets
from GUI.GUI import MyWindow
import sys

# 主程序
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec_())
