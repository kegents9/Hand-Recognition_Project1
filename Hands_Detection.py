import cv2
import mediapipe as mp
import time
import math
import numpy as np

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# while True:
#     success, img = cap.read()
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = mp_hands.process(imgRGB)
#     xList = []
#     yList = []
#     lmList = []
    
#     # print(results.multi_hand_landmarks)
#     for handLms in results.multi_hand_landmarks:
#         print(handLms)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
with mp_hands.Hands(
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75) as hands:
  if cap.isOpened():
    success, image = cap.read()
    image_size=image.shape
    image_height=image_size[0]
    image_width=image_size[1]
    print('size', image_size, '\n', 'height', image_height, '\n', 'width', image_width)

  while cap.isOpened():
    success, image = cap.read()

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # print(len(results))

    # cv2.putText(img, text, (org), fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    hint = 'press esc to leave'
    imgx, imgy, imgw, imgh = 0, 0, int(image_width*0.475), int(image_height*0.08)
    cv2.rectangle(image, (imgx, imgx), (imgx + imgw, imgy + imgh), (0,0,0), -1)
    cv2.putText(image,hint,  (0,int(image_height*0.06)), 
                cv2.FONT_HERSHEY_SIMPLEX, 1,
          (0,95,255),2,cv2.LINE_AA)

    if results.multi_hand_landmarks:
    #   print('**********************************************************')
    #   print(len(results.multi_hand_landmarks))
    #   print('**********************************************************')
      
      for hand_landmarks in results.multi_hand_landmarks:
        THUMB_TIP_X = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)
        THUMB_TIP_Y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

        INDEX_TIP_X = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
        INDEX_TIP_Y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)

        MID_TIP_X = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)
        MID_TIP_Y = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

        RING_TIP_X = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)
        RING_TIP_Y = int(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

        PINKY_TIP_X = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)
        PINKY_TIP_Y = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)

        # POSITION OF THUMB      
        p_T=np.array([THUMB_TIP_X,THUMB_TIP_Y])
        # POSITION OF INDEX FINGER
        p_I=np.array([INDEX_TIP_X,INDEX_TIP_Y])
        # POSITION OF MIDDLE FINGER
        p_M=np.array([MID_TIP_X,MID_TIP_Y])
        # POSITION OF RING FINGER
        p_R=np.array([RING_TIP_X,RING_TIP_Y])
        # POSITION OF PINKY
        p_P=np.array([PINKY_TIP_X,PINKY_TIP_Y])
        
        # DISTANCE FROM THUMB TO INDEX FINGER
        D_T_I = p_T-p_I
        DTI = int(math.hypot(D_T_I[0],D_T_I[1]))

        # DISTANCE FROM THUMB TO MIDDLE FINGER
        D_T_M = p_T-p_M
        DTM = int(math.hypot(D_T_M[0],D_T_M[1]))

        # DISTANCE FROM THUMB TO RING FINGER
        D_T_R = p_T-p_R
        DTR = int(math.hypot(D_T_R[0],D_T_R[1]))

        # DISTANCE FROM THUMB TO PINKY FINGER
        D_T_P = p_T-p_P
        DTP = int(math.hypot(D_T_P[0],D_T_P[1]))

        if (DTP<=80) and (DTR<=80) and (DTI>=80) and (DTM<=60):
          V = 1
        elif (DTP<=80) and (DTR<=100) and (DTI>=100) and (DTM>=100):
          V = 2
        elif (DTP<=50) and (DTR>=100) and (DTI>=100) and (DTM>=100):
          V = 3
        elif (DTP>=50) and (DTR>=100) and (DTI>=100) and (DTM>=100):
          if (DTP<180) and (DTM>DTI):
            V = 4
          elif (DTP>=180):
            V = 5
        else:
          V = 0

        if(V==1):
          text = 'one'
          cv2.putText(image,
          text,
          (int(image_width*0.75),int(image_height*0.08)),cv2.FONT_HERSHEY_SIMPLEX,1,
          (0,255,0),6,cv2.LINE_AA)

          cv2.putText(image,
          text,
          (int(image_width*0.75),int(image_height*0.08)),cv2.FONT_HERSHEY_SIMPLEX,1,
          (0,0,0),2,cv2.LINE_AA)
        elif(V==2):
          text = 'two'
          cv2.putText(image,
          text,
          (int(image_width*0.75),int(image_height*0.08)),cv2.FONT_HERSHEY_SIMPLEX,1,
          (0,255,0),6,cv2.LINE_AA)

          cv2.putText(image,
          text,
          (int(image_width*0.75),int(image_height*0.08)),cv2.FONT_HERSHEY_SIMPLEX,1,
          (0,0,0),2,cv2.LINE_AA)
        elif(V==3):
          text = 'three'
          cv2.putText(image,
          text,
          (int(image_width*0.75),int(image_height*0.08)),cv2.FONT_HERSHEY_SIMPLEX,1,
          (0,255,0),6,cv2.LINE_AA)

          cv2.putText(image,
          text,
          (int(image_width*0.75),int(image_height*0.08)),cv2.FONT_HERSHEY_SIMPLEX,1,
          (0,0,0),2,cv2.LINE_AA)
        elif(V==4):
          text = 'four'
          cv2.putText(image,
          text,
          (int(image_width*0.75),int(image_height*0.08)),cv2.FONT_HERSHEY_SIMPLEX,1,
          (0,255,0),6,cv2.LINE_AA)

          cv2.putText(image,
          text,
          (int(image_width*0.75),int(image_height*0.08)),cv2.FONT_HERSHEY_SIMPLEX,1,
          (0,0,0),2,cv2.LINE_AA)
        elif(V==5):
          text = 'five'
          cv2.putText(image,
          text,
          (int(image_width*0.75),int(image_height*0.08)),cv2.FONT_HERSHEY_SIMPLEX,1,
          (0,255,0),6,cv2.LINE_AA)

          cv2.putText(image,
          text,
          (int(image_width*0.75),int(image_height*0.08)),cv2.FONT_HERSHEY_SIMPLEX,1,
          (0,0,0),2,cv2.LINE_AA)

        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # press esc to leave
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()