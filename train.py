import gzip

import numpy as np
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

with gzip.open('train-labels-idx1-ubyte.gz') as train_labels:
    data_from_train_file = train_labels.read()

# пропускаем первые 8 байт
label_data = data_from_train_file[8:]
assert len(label_data) == 60000

# конвертируем каждый байт в целое число.
# это будет число от 0 до 9
labels = [int(label_byte) for label_byte in label_data]
assert min(labels) == 0 and max(labels) == 9
assert len(labels) == 60000

SIZE_OF_ONE_IMAGE = 28 ** 2  # размер картинки - 28 на 28 пикселей
images = []

# перебор тренировочного файла и чтение одного изображения за раз
with gzip.open('train-images-idx3-ubyte.gz') as train_images:
    train_images.read(4 * 4)
    ctr = 0
    for _ in range(60000):
        image = train_images.read(size=SIZE_OF_ONE_IMAGE)
        assert len(image) == SIZE_OF_ONE_IMAGE

        # Конвертировать в NumPy
        image_np = np.frombuffer(image, dtype='uint8') / 255
        images.append(image_np)

images = np.array(images)

labels_np = np.array(labels).reshape((-1, 1))

encoder = OneHotEncoder(categories='auto')
labels_np_onehot = encoder.fit_transform(labels_np).toarray()

X_train, X_test, y_train, y_test = train_test_split(images, labels_np_onehot)

# создаём модель свёрточной нейросети
model = keras.Sequential()
model.add(keras.layers.Dense(input_shape=(SIZE_OF_ONE_IMAGE,), units=128, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))


model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# обучаем нейросеть
model.fit(X_train, y_train, epochs=10, batch_size=128)

# тестируем нейросеть
model.evaluate(X_test, y_test)

model.save('mnist.h5')
