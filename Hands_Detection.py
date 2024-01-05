import cv2
import mediapipe as mp
import math
import numpy as np

# Constants
ESC_KEY = 27
TEXT_POSITION_SCALE = 0.75
TEXT_HEIGHT_SCALE = 0.08
TEXT_COLOR = (0, 0, 0)
TEXT_HIGHLIGHT = (0, 255, 0)
TEXT_THICKNESS = 2
TEXT_HIGHLIGHT_THICKNESS = 6
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_FONT_SCALE = 1

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return int(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))

# Function to determine the gesture value
def get_gesture_value(DTI, DTM, DTR, DTP):
    return (
        1 if (DTP <= 80) and (DTR <= 80) and (DTI >= 80) and (DTM <= 60) else
        2 if (DTP <= 80) and (DTR <= 100) and (DTI >= 100) and (DTM >= 100) else
        3 if (DTP <= 50) and (DTR >= 100) and (DTI >= 100) and (DTM >= 100) else
        4 if (DTP >= 50) and (DTR >= 100) and (DTI >= 100) and (DTM >= 100) and (DTP < 180) and (DTM > DTI) else
        5 if (DTP >= 180) else
        0
    )

# Gesture to text mapping
gesture_text = {
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five'
}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

with mp_hands.Hands(
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark positions and calculate distances
                landmarks = hand_landmarks.landmark
                thumb_tip = np.array([landmarks[mp_hands.HandLandmark.THUMB_TIP].x * image_width,
                                      landmarks[mp_hands.HandLandmark.THUMB_TIP].y * image_height])
                index_tip = np.array([landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                                      landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height])
                middle_tip = np.array([landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width,
                                       landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height])
                ring_tip = np.array([landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width,
                                     landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height])
                pinky_tip = np.array([landmarks[mp_hands.HandLandmark.PINKY_TIP].x * image_width,
                                      landmarks[mp_hands.HandLandmark.PINKY_TIP].y * image_height])

                DTI = calculate_distance(thumb_tip, index_tip)
                DTM = calculate_distance(thumb_tip, middle_tip)
                DTR = calculate_distance(thumb_tip, ring_tip)
                DTP = calculate_distance(thumb_tip, pinky_tip)

                # Determine gesture value
                V = get_gesture_value(DTI, DTM, DTR, DTP)

                # Draw text based on gesture value
                if V in gesture_text:
                    text = gesture_text[V]
                    text_position = (int(image_width * TEXT_POSITION_SCALE), int(image_height * TEXT_HEIGHT_SCALE))
                    cv2.putText(image, text, text_position, TEXT_FONT, TEXT_FONT_SCALE, TEXT_HIGHLIGHT, TEXT_HIGHLIGHT_THICKNESS, cv2.LINE_AA)
                    cv2.putText(image, text, text_position, TEXT_FONT, TEXT_FONT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)

                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == ESC_KEY:
            break

cap.release()
