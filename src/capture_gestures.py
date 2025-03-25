import cv2
import os

def capture_images(gesture_name, num_images=50):
    save_dir = f"../dataset/{gesture_name}"
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print(f"Capturing images for gesture: {gesture_name}. Press 'q' to quit.")

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture Gesture", frame)

        # Save image when 's' is pressed
        cv2.imwrite(f"{save_dir}/{count}.jpg", frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {count} images for gesture: {gesture_name}")

if __name__ == "__main__":
    gesture_name = input("Enter gesture name (Hello, Hi, Bye, etc.): ")
    capture_images(gesture_name)
