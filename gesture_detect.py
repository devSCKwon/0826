import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)

# rps_gesture = ['ddadong', 'ok', 'victory']##################
detector = HandDetector(maxHands=1, detectionCon=0.7)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hands, frame = detector.findHands(frame, flipType=False)
    if hands:
        lm_list = hands[0]['lmList']
        lm_list = np.array(lm_list)

        v1 = lm_list[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
        v2 = lm_list[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
        v = v2 - v1  # (20,3) 팔목과 각 손가락 관절 사이의 벡터를 구한다.
        ab_v = np.linalg.norm(v, axis=1)
        print(v)
        print(ab_v)


    cv2.imshow('gesture_detect', frame)

    if cv2.waitKey(1) == 27:
        break