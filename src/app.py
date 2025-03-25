import os

print("1. Capture Gesture Images")
print("2. Train Model")
print("3. Recognize Gesture")
choice = input("Enter your choice: ")

if choice == "1":
    os.system("python capture_gestures.py")
elif choice == "2":
    os.system("python train_model.py")
elif choice == "3":
    os.system("python recognize_gesture.py")
else:
    print("Invalid Choice")
