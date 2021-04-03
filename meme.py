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

model = keras.models.load_model('mnist.h5')


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

        self.labels = []
        for i in range(0, 10):
            label = QLabel(self)
            label.setGeometry(510, 10 + 37 * i, 180, 39)
            label.setText(str(i) + ": ")
            label.setAlignment(Qt.AlignLeft)
            font = label.font()
            font.setPointSize(28)
            label.setFont(font)
            self.labels.append(label)

        # making image color to white
        self.image.fill(Qt.white)

        # variables
        # drawing flag
        self.drawing = False
        # default brush size
        self.brushSize = 50
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
            self.predict()

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
        global model

        def plot_image(pixels: np.array, width=28, height=28):
            plt.imshow(pixels.reshape((width, height)), cmap='gray')
            plt.show()

        pic = ImageQt.fromqimage(self.image)
        pic = pic.resize((28, 28))
        pixels = np.asarray(pic.getdata())

        pixels = np.array([round(255 - sum(e) / len(e)) for e in pixels])
        # plot_image(pixels)

        left = 28
        top = 28
        right = 0
        bottom = 0


        for y in range(0, 28):
            for x in range(0, 28):
                index = y * 28 + x
                pixel = pixels[index]
                is_black = pixel == 0
                if not is_black:
                    left = min(left, x)
                    top = min(top, y)
                    right = max(right, x)
                    bottom = max(bottom, y)
                print(f'{pixels[index]:4}', end='')
            print()
        right += 1
        bottom += 1
        width = right - left
        height = bottom - top

        print(f"left: {left}, top: {top}")
        print(f"right: {right}, bottom: {bottom}")
        print(f"width: {width}, height: {height}")

        pic = pic.crop((left, top, right, bottom))

        content = np.asarray(pic.getdata())
        content = np.array([round(255 - sum(e) / len(e)) for e in content])
        # plot_image(content, height, width)

        pixels_matrix = np.array([[0] * 28] * 28)
        print(pixels_matrix.shape)
        x = (28 - width) // 2
        y = (28 - height) // 2

        pixels_matrix[y:(y + height), x:(x+width)] = content.reshape(height, width)

        plot_image(pixels_matrix)

        image_to_predict = pixels_matrix

        predicted_results = model.predict(image_to_predict.reshape((1, -1)))
        probabilities = list(predicted_results[0])
        # print('\n'.join([str(i) + ': ' + str(round(e * 100)) + '%' for i, e in enumerate(probabilities)]))
        probabilities = [round(e * 100)for i, e in enumerate(probabilities)]
        result = probabilities.index(max(probabilities))
        for index, label in enumerate(self.labels):
            label.setText(str(index) + ": " + str(probabilities[index]))
            if result == index:
                label.setStyleSheet("color: red;")
            else:
                label.setStyleSheet("color: black;")

# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
window = Window()

# showing the wwindow
window.show()

# start the app
sys.exit(App.exec())
