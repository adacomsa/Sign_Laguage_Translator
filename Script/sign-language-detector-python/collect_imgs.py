import os
import cv2

DATA_DIR = 'C:\\Users\\adacomputer\\Desktop\\Licenta\\Classes'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Automatically detect available camera index
camera_index = -1
for index in range(10):  # Check the first 10 indices as a reasonable range
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        camera_index = index
        cap.release()
        break
if camera_index == -1:
    print("No available camera detected.")
    exit()

cap = cv2.VideoCapture(camera_index)
number_of_classes = 19
dataset_size = 100

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Check your camera.")
            break
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Check your camera.")
            break
        cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("Data collection stopped early.")
            break

cap.release()
cv2.destroyAllWindows()
