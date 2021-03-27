# importing libraries
import sys

import PyQt5
from PIL import Image
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras as keras
from PIL import Image, ImageQt


# window class
class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # setting title
        self.setWindowTitle("Paint with PyQt5")

        # setting geometry to main window
        self.setGeometry(100, 100, 700, 500)
        self.rect2 = QRect(0, 0, 500, 500)

        # creating button
        self.button = QPushButton(self)
        self.button.setGeometry(510, 440, 180, 30)
        self.button.setText('Predict')
        self.button.clicked.connect(self.predict)

        # creating button
        self.clearButton = QPushButton(self)
        self.clearButton.setGeometry(510, 400, 180, 30)
        self.clearButton.setText('Clear')
        self.clearButton.clicked.connect(self.clear)

        # creating image object
        self.image = QImage(QSize(500, 500), QImage.Format_RGB32)
        print(str(self.rect()))

        #
        self.label = QLabel(self)
        self.label.setGeometry(510, 10, 180, 390)
        self.label.setText("Ы")
        self.label.setAlignment(Qt.AlignCenter)
        font = self.label.font()
        font.setPointSize(50)
        font.setBold(True)
        self.label.setFont(font)

        # making image color to white
        self.image.fill(Qt.white)

        # variables
        # drawing flag
        self.drawing = False
        # default brush size
        self.brushSize = 32
        # default color
        self.brushColor = Qt.black

        # QPoint object to tract the point
        self.lastPoint = QPoint()


    # method for checking mouse cicks
    def mousePressEvent(self, event):

        # if left mouse button is pressed
        if event.button() == Qt.LeftButton:
            # make drawing flag true
            self.drawing = True
            # make last point to the point of cursor
            self.lastPoint = event.pos()

    # method for tracking mouse activity
    def mouseMoveEvent(self, event):

        # checking if left button is pressed and drawing flag is true
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            # creating painter object
            painter = QPainter(self.image)

            # set the pen of the painter
            painter.setPen(QPen(self.brushColor, self.brushSize,
                                Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

            # draw line from the last point of cursor to the current point
            # this will draw only one step
            painter.drawLine(self.lastPoint, event.pos())

            # change the last point
            self.lastPoint = event.pos()
            # update
            self.update()

    # method for mouse left button release
    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:
            # make drawing flag false
            self.drawing = False

    # paint event
    def paintEvent(self, event):
        # create a canvas
        canvasPainter = QPainter(self)

        # draw rectangle  on the canvas
        canvasPainter.drawImage(self.rect2, self.image, self.image.rect())

    # method for saving canvas
    def save(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "",
                                                  "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")

        if filePath == "":
            return
        self.image.save(filePath)

    # method for clearing every thing on canvas
    def clear(self):
        # make the whole canvas white
        self.image.fill(Qt.white)
        # update
        self.update()

    def predict(self):
        pic = ImageQt.fromqimage(self.image)
        pic = pic.resize((28, 28), Image.ANTIALIAS)
        pixels = np.asarray(pic.getdata())
        pixels = np.array([round(255 - sum(e) / len(e)) for e in pixels])


        image_to_predict = pixels

        model = keras.models.load_model('mnist.h5')
        predicted_results = model.predict(image_to_predict.reshape((1, -1)))
        probabilities = list(predicted_results[0])
        print('\n'.join([str(i) + ': ' + str(round(e * 100)) + '%' for i, e in enumerate(probabilities)]))
        result = probabilities.index(max(probabilities))
        self.label.setText(str(result))


# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

# showing the wwindow
window.show()

# start the app
sys.exit(App.exec())
