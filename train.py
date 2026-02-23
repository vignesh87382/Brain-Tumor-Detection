import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Parameters
IMG_SIZE = 128
DATA_DIR = "dataset"

categories = ["no", "yes"]

data = []
labels = []

for category in categories:
    path = os.path.join(DATA_DIR, category)
    class_num = categories.index(category)

    if not os.path.exists(path):
        continue

    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(img_array)
            labels.append(class_num)
        except:
            pass

data = np.array(data) / 255.0
labels = np.array(labels)

if len(data) == 0:
    print("No images found. Add dataset inside dataset/yes and dataset/no folders.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

model.save("brain_tumor_model.h5")

print("Model saved successfully.")
