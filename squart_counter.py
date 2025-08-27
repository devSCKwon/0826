import cv2
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(0)
detector = PoseDetector(detectionCon=0.7)
cnt = 0
flag = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = detector.findPose(frame)
    lmList, bboxInfo = detector.findPosition(frame)

    if lmList:
        angle0, frame = detector.findAngle(lmList[23], lmList[25], lmList[27], img=frame, color = (0,0,255), scale=10)
        # print(angle0)

        angle1, frame = detector.findAngle(lmList[24], lmList[26], lmList[28], img=frame, color=(0, 255, 0), scale=10)
        # print(angle1)

        if angle0 < 90 and angle1 < 90 and flag == 0 :
            cnt = cnt + 1
            flag = 1
        elif angle0 > 150 and angle1 > 150 and flag == 1:
            flag = 0

    # if lmList:
    #     cv2.putText(frame, lmList, (bboxInfo[0][0][0], bboxInfo[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if lmList:
        # 스쿼트 카운트를 화면에 표시
        cv2.putText(frame, f'Squat Count: {cnt}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # 각도 정보도 함께 표시 (선택사항)
        cv2.putText(frame, f'Left Knee: {int(angle0)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f'Right Knee: {int(angle1)}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('squart_counter', frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()