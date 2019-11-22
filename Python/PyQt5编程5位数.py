import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QAction, QMessageBox
import random
from PyQt5.QtCore import pyqtSlot
from PyQt5.Qt import QLineEdit

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = '随机数'
        self.left = 300
        self.top = 300
        self.width = 320
        self.height = 200
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # create textbox
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(280, 40)

        # Create a button in the window
        self.button = QPushButton('确定', self)
        self.button.move(20, 80)

        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        self.show()

    @pyqtSlot()
    def on_click(self):
        textboxValue = self.textbox.text()
        number = textboxValue.split(' ')
        while True:
            number = list(set(number))
            if len(number) <= 4:
                number.append(random.randrange(1,10))
            elif len(list(set(number))) == 5:
                break
        QMessageBox.question(self, "Message", 'You typed:' + str(number),
                             QMessageBox.Ok, QMessageBox.Ok)
        self.textbox.setText(str(number).replace('\'','').replace('[','').replace(']','').replace(' ',''))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    app.exit(app.exec_())

