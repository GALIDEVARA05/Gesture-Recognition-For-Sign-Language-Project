import cv2
import numpy as np
import tensorflow as tf
import os

# Load the trained model
model = tf.keras.models.load_model("../models/gesture_model.h5")

# Automatically fetch gesture names from the dataset
gesture_dataset_path = "../dataset"

labels = sorted(os.listdir(gesture_dataset_path))  # Dynamically load labels
print(f"Loaded labels: {labels}")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image (resize and normalize)
    img = cv2.resize(frame, (64, 64))
    img_array = np.expand_dims(img / 255.0, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    gesture = labels[np.argmax(prediction)]

    # Display the recognized gesture
    cv2.putText(frame, f"Gesture: {gesture}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
