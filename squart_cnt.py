import cv2
from cvzone.PoseModule import PoseDetector
cap = cv2.VideoCapture(0)
detector = PoseDetector(detectionCon=0.7)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow('squart_counter', frame)

    if cv2.waitKey(1) == 27:
        break