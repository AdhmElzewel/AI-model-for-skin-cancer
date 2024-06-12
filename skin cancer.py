import cv2
import numpy as np
import os
datadir = r"C:\Users\20114\Desktop\Downloads\skin cancer\train"
categories = ["benign", "malignant"]
training_data = []
image_size = 60
for category in categories:
    path = os.path.join(datadir, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (image_size, image_size))
            training_data.append([new_array, class_num])
        except Exception as ex:
            pass
print(len(training_data))
a = []
b = []

for features1, labels1 in training_data:
    a.append(features1)
    b.append(labels1)
a = np.array(a)
b = np.array(b)
datadir_test = r"C:\Users\20114\Desktop\Downloads\skin cancer\test"
categories_test = ["benign", "malignant"]
testing_data = []
for category1 in categories_test:
    path1 = os.path.join(datadir_test, category1)
    class_num1 = categories_test.index(category1)
    for img1 in os.listdir(path1):
        try:
            img_array1 = cv2.imread(os.path.join(path1, img1))
            new_array1 = cv2.resize(img_array1, (image_size, image_size))
            testing_data.append([new_array1, class_num1])
        except Exception as ex:
            pass
print(len(testing_data))
a_test = []
b_test = []
for features, labels in testing_data:
    a_test.append(features)
    b_test.append(labels)
a_test = np.array(a)
b_test = np.array(b)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras import regularizers
batch = 10
epochs = 10
validation_split = 0.2
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(image_size, image_size, 3), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(a, b, batch_size=batch, epochs=epochs)
b_p = (model.predict(a_test) > 0.5).astype(int)
from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(b_test, b_p))
print("Precision:", metrics.precision_score(b_test, b_p))
print("Recall:", metrics.recall_score(b_test, b_p))
print("confusion mat:", metrics.confusion_matrix(b_test, b_p))
