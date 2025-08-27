
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np



#rps_gesture = {0: 'rock', 1: 'scissor', 2: 'paper'}
#rps_gesture = {0: 'one', 1: 'two', 2: 'three'}
rps_gesture = ['ddadong', 'ok', 'victory']##################

detector = HandDetector(maxHands = 1,detectionCon=0.75)
#file = np.genfromtxt('aaaaa.csv', delimiter=',')  # 파일을 읽어온다
file = np.genfromtxt('motion.csv', delimiter=',')  # 파일을 읽어온다
angle = file[:, :-1].astype(np.float32)  # 0번인덱스 부터 마지막 인덱스(-1) 전까지 잘라라
label = file[:, -1].astype(np.float32)  # 마지막 인덱스(-1)만 가져와라

knn = cv2.ml.KNearest_create()  # knn모델을 초기화
knn.train(angle, cv2.ml.ROW_SAMPLE, label)  # knn 학습

cap_cam = cv2.VideoCapture(0)

flag = 1
while cap_cam.isOpened():  # 카메라가 열려 있으면
    cam_ret, cam_img = cap_cam.read()
    cam_img = cv2.flip(cam_img, 1)

    if not cam_ret:
        break
    hands, cam_img = detector.findHands(cam_img, flipType=False)

    if hands:


        lm_list = hands[0]['lmList']
        lm_list = np.array(lm_list)

        v1 = lm_list[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
        #print(v1)
        v2 = lm_list[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
        v = v2 - v1  # (20,3) 팔목과 각 손가락 관절 사이의 벡터를 구한다.
        print(v)
        xx = np.array([[1, 2, 3],
                      [4, 5, 6]])
        ex = np.linalg.norm(xx, axis=1)
        print(xx.shape)## 20행 3열
        ab_v=np.linalg.norm(v, axis=1)
        print(v.shape)
        #print(ab_v.shape)
        v = v / np.expand_dims(ab_v, axis=-1)  # 유닛벡터 구하기 벡터/벡터의 길이

        angle = np.arccos(np.einsum('nt,nt->n',
                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                    :]))  # [15,] 유닛벡터를 내적한 값의 아크 코사인을 구하면 각도를 구할 수 있다.
        angle = np.degrees(angle)  # Convert radian to degree

        angle = np.expand_dims(angle.astype(np.float32),
                               axis=0)  # float32 차원증가 keras or tensor 머신러닝 모델에 넣어서 추론할 때는 항상 맨앞 차원 하나를 추가한다.
        _, results, _, _ = knn.findNearest(angle, 3)  # statue,result,인접값,거리
        # print(results)
        idx = int(results[0][0])
        gesture_name = rps_gesture[idx]

        # print(gesture_name)
        cv2.putText(cam_img, text=gesture_name, org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2,
                    color=(255, 255, 255), thickness=2)


    cv2.imshow('cam', cam_img)

    if cv2.waitKey(1) == ord('q'):
        break