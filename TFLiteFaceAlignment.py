#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import cv2
import tensorflow as tf
from functools import partial
import time

from tensorflow._api.v2 import image
from TFLiteFaceDetector import UltraLightFaceDetecion
import sys

class CoordinateAlignmentModel():


 if __name__ == '__main__':
    fd = UltraLightFaceDetecion("weights/RFB-320.tflite",conf_threshold=0.88)
    fa = CoordinateAlignmentModel("OpenVtuber-master\coor_2d106.tflite")

    img = cv2.imread("۲۰۲۱۰۵۰۴_۱۴۵۲۰۳.jpg")
    # img = cv2.resize(img , (0,0),fx=0.5 , fy = 0.5)
    color = (125, 255, 125)

   
    boxes, scores = fd.inference(img)

    for pred in fa.get_landmarks(img, boxes):
        for p , index in np.round(pred).astype(np.int):
            print(p,index)
            cv2.circle(img, tuple(p), 1, color, 1, cv2.LINE_AA)
            cv2.putText(img , str(index), p ,cv2.FONT_HERSHEY_SIMPLEX, 0,25,(0,0,255),0)
    
    cv2.imwrite("yasamin")
    cv2.imshow("result", img)
    cv2.waitKey()
           
