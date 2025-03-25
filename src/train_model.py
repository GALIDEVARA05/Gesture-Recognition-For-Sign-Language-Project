import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

def load_data():
    X, y = [], []
    labels = os.listdir("../dataset")
    label_map = {label: i for i, label in enumerate(labels)}

    for label in labels:
        path = f"../dataset/{label}"
        for img_file in os.listdir(path):
            img_path = os.path.join(path, img_file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))
            X.append(img)
            y.append(label_map[label])

    X = np.array(X) / 255.0
    y = tf.keras.utils.to_categorical(np.array(y), len(labels))
    return X, y, label_map

def build_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X, y, label_map = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_model(len(label_map))
    model.summary()

    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    model.save("../models/gesture_model.h5")
    print("Model training completed and saved.")
