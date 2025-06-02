import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)  # Adjust camera index
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

labels_dict = {0: 'House', 1: 'OK', 2: 'OK', 3: 'I love you', 4: 'I love you', 5: 'Book',
               6: 'Bed', 7: 'Bed', 8: ' Sit down', 9: 'Sit down', 10: 'Angry', 11: 'Love',
               12: 'Love', 13: 'Stop', 14: 'Stop', 15: 'Water', 16: 'Water', 17: 'Friend', 18: 'Friend'}  # Update labels as needed
confidence_threshold = 0.7  # Define a threshold for prediction confidence

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux, x_, y_ = [], [], []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )

            for point in hand_landmarks.landmark:
                x_.append(point.x)
                y_.append(point.y)
            if x_ and y_:
                min_x, min_y = min(x_), min(y_)
                for point in hand_landmarks.landmark:
                    data_aux.extend([point.x - min_x, point.y - min_y])

        # Ensure consistent number of landmarks (21 points * 2 = 42 features for a single hand)
        while len(data_aux) < 84:
            data_aux.extend([0.0, 0.0])  # Pad with zeros if fewer landmarks are detected

        x1 = max(0, int(min(x_) * W) - 10)
        y1 = max(0, int(min(y_) * H) - 10)
        x2 = min(W, int(max(x_) * W) - 10)
        y2 = min(H, int(max(y_) * H) - 10)

        # Get the prediction and confidence
        prediction = model.predict([np.asarray(data_aux[:84])])
        prediction_proba = model.predict_proba([np.asarray(data_aux[:84])])  # Get probabilities

        predicted_class = int(prediction[0])
        predicted_confidence = max(prediction_proba[0])  # Confidence of the predicted class

        # If confidence is below threshold, classify as 'Other'
        if predicted_confidence < confidence_threshold:
            predicted_character = 'Unknown sign'
        else:
            predicted_character = labels_dict.get(predicted_class, 'Unknown sign')

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
