import pandas as pd
df = pd.read_csv(r"C:\Users\20114\Desktop\Downloads\iris\Iris.csv")
a = df.iloc[:, 0:4]
b = df.iloc[:, 4]
from sklearn.model_selection import train_test_split
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=0)
from keras.utils import to_categorical
dummy_b = to_categorical(b_train)
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
batch = 1
epochs = 50
validation = 0.2
model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(4,)))
model.add(Dense(4, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(a_train, dummy_b, batch_size=batch, epochs=epochs)
b_p = model.predict(a_test)
