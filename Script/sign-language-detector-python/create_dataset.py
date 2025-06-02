import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

DATA_DIR = 'C:\\Users\\adacomputer\\Desktop\\Licenta\\Classes'

data = []
labels = []

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

for dir_ in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_dir):
        continue  # Skip non-directory files
    for img_path in os.listdir(class_dir):
        img_full_path = os.path.join(class_dir, img_path)
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Warning: Skipping {img_full_path} (failed to read image)")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for point in hand_landmarks.landmark:
                    x_.append(point.x)
                    y_.append(point.y)

                if x_ and y_:
                    min_x, min_y = min(x_), min(y_)
                    for point in hand_landmarks.landmark:
                        data_aux.append(point.x - min_x)
                        data_aux.append(point.y - min_y)

            data.append(data_aux)
            labels.append(int(dir_))  # Assuming class labels are numeric

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
