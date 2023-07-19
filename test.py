import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import tensorflow
import math
import time
cap=cv2.VideoCapture(0)

detector= HandDetector(maxHands=1)
classifier=Classifier("model/keras_model.h5","model/labels.txt")

counter=0
folder="image/e"

labels=["A","B","C","D","E"]
while True:
    success, img=cap.read()
    imgout=img.copy()
    hands,img=detector.findHands(img)
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        imgwhite=np.ones((300,300,3),np.uint8)*255
        imgcrop=img[y-20:y+h+20,x-20:x+w+20]
        
        aspectratio=h/w
        if aspectratio>1:
            k=math.ceil(300/h*w)
            resize=cv2.resize(imgcrop,(k,300))
            gap=math.ceil((300-k)/2)
            imgwhite[:, gap:gap+k]=resize
            pred,idx=classifier.getPrediction(imgwhite)
            print(pred,idx)
        else:
            k=math.ceil(300/w*h)
            resize=cv2.resize(imgcrop,(300,k))
            gap=math.ceil((300-k)/2)
            imgwhite[gap:gap+k,:]=resize
            pred,idx=classifier.getPrediction(imgwhite)
        cv2.putText(imgout,labels[idx],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        cv2.rectangle(imgout,(x-20,y-20),(x+w+20,x+h+20),(255,0,255),4)

        cv2.imshow("imagecrop",imgcrop)
        cv2.imshow("imagewhite",imgwhite)
    cv2.imshow("Image", imgout)
    cv2.waitKey(1)