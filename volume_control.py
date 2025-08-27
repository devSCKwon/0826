from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities ,IAudioEndpointVolume
devices=AudioUtilities.GetSpeakers()
#import pyautogui
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume=cast(interface,POINTER(IAudioEndpointVolume))
volRange=volume.GetVolumeRange()
minVol=volRange[0]
maxVol=volRange[1]
# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.7)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hands, frame = detector.findHands(frame)
    if hands:
        lmList = hands[0]['lmList']
        length, info, frame = detector.findDistance(lmList[4], lmList[8], frame)
        if length < 30:
            fqy = lmList[4][1]/480.0
            if fqy > 1 : fqy = 1
            elif fqy < 0 : fqy = 0

            vol = np.interp(fqy, [0, 1], [maxVol, minVol])
            volume.SetMasterVolumeLevel(vol, None)  # 0:max~ -65:0

        print(f"Length : {length}")
    cv2.imshow('volume_control', frame)

    if cv2.waitKey(1) == 27:
        break
