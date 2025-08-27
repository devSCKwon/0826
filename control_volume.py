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


# Initialize the HandDetector class with the given parameters
detector = HandDetector(maxHands = 1,detectionCon=0.75)

cap_cam =cv2.VideoCapture(0)
w=int(cap_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
h=int(cap_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
while cap_cam.isOpened():#카메라가 열려 있으면
    cam_ret,cam_img=cap_cam.read()
    cam_img = cv2.flip(cam_img, 1)
    #video_ret, video_img = cap_video.read()

    if not cam_ret:
        break
    hands,cam_img=detector.findHands(cam_img,flipType=False)
    if hands:
        lm_list=hands[0]['lmList']
        length, info, cam_img = detector.findDistance(lm_list[4], lm_list[8], cam_img)
        if length <50:
            rel_x = lm_list[4][1] / h
            if rel_x>1 : rel_x=1
            elif rel_x<0 : rel_x=0
            #print(68*rel_x-68)
            vol=np.interp(rel_x,[0,1],[maxVol,minVol])
            volume.SetMasterVolumeLevel(vol, None)#0:max~ -65:0
            #volume.SetMasterVolumeLevel(-28, None)  # 0:max~ -28:0

    cv2.imshow('cam', cam_img)

    if cv2.waitKey(1) == ord('q'):
        break


