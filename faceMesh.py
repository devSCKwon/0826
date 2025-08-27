import cv2
import pygame
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = FaceMeshDetector(maxFaces=1, minDetectionCon=0.7)

cnt = 0
pygame.init()
pygame.mixer_music.load('alram.mp3')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    # frame = cv2.resize(frame, (0,0), fx=2.0, fy=2.0)
    frame, faces = detector.findFaceMesh(frame)
    if faces:
        lvd, _ = detector.findDistance(faces[0][159], faces[0][145])
        lhd, _ = detector.findDistance(faces[0][33], faces[0][133])
        left_ratio = lvd/lhd
        print(lvd, lhd, left_ratio, cnt)

        if left_ratio < 0.3 :
            cnt = cnt+1
        else:
            cnt = cnt-1
            if cnt < 0: cnt = 0
            # print(lvd, lhd, left_ratio, cnt)

        if cnt > 100 and pygame.mixer_music.get_busy() == 0:
            pygame.mixer_music.play()
            print("A")

        for i in range(len(faces[0])):
            cv2.putText(frame, str(i), (faces[0][i][0]+10, faces[0][i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)

    cv2.imshow('faceMesh', frame)

    if cv2.waitKey(1) == 27:
        break