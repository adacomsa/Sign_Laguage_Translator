# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2  # Import OpenCV
import numpy as np

# Define hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
]

# Helper function to draw landmarks and connections on the image
def draw_landmarks_on_image(image, detection_result):
    annotated_image = image.copy()
    # Iterate through each detected hand
    for hand_landmarks in detection_result.hand_landmarks:
        # Draw connections between landmarks
        for start_idx, end_idx in HAND_CONNECTIONS:
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            x_start = int(start.x * image.shape[1])
            y_start = int(start.y * image.shape[0])
            x_end = int(end.x * image.shape[1])
            y_end = int(end.y * image.shape[0])
            cv2.line(annotated_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in hand_landmarks:  # Access each landmark in the hand
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(annotated_image, (x, y), 5, (0, 0, 255), -1)
    return annotated_image

# STEP 2: Create a HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Start capturing video from the webcam.
cap = cv2.VideoCapture(0)  # 0 is the default camera. Use 1 or 2 for external cameras.

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        ret, frame = cap.read()  # Capture frame-by-frame
        if not ret:
            print("Error: Unable to read from camera.")
            break

        # Convert the frame to RGB for Mediapipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect hand landmarks
        detection_result = detector.detect(image)

        # Convert the image back to a numpy array for visualization
        frame_np = image.numpy_view()

        # Draw landmarks and connections on the frame
        annotated_frame = draw_landmarks_on_image(frame_np, detection_result)

        # Display the processed frame
        cv2.imshow("Hand Landmarks", annotated_frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
