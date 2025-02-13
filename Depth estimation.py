import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt

#functions
import HSV_filter as hsv
import shape_recognition as shape
import triangulation as tri

#open both cameras
cap_right = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap_left = cv2.VideoCapture(1,cv2.CAP_DSHOW)

frame_rate = 120

Baseline = 9
F = 6
alpha = 56.6

#Initial values
count = -1

while(True):
    count+=1

    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read()


    if ret_right==False or ret_left==False:
       break
    
    else:
       mask_right = hsv.add_HSV_filter(frame_right,1)
       mask_left = hsv.add_HSV_Filter(frame_left,0)

       res_right = cv2.bitwise_and(frame_right, frame_right, mask=mask_right)
       res_left = cv2.bitwise_and(frame_left, frame_left, mask=mask_left)

       circles_right = shape.find_circles(frame_right, mask_right)
       circles_left = shape.find_circles(frame_left, mask_left)

       if np.all(circles_right) == None or np.all(circles_left) == None:
           cv2.putText(frame_right, "TRACKING LOST",(75,50), cv2.FONT_HERSHEY_SIMPLEX,0)
           cv2.putText(frame_left, "TRACKING LOST",(75,50), cv2.FONT_HERSHEY_SIMPLEX,0)
       else:
           depth = tri.find_depth(circles_right, circles_left, frame_right, frame_left,Baseline,alpha)

           cv2.putText(frame_right, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(124,252,0),2)
           cv2.putText(frame_left, "TRACKING", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(124,252,0),2)
           cv2.putText(frame_right, "Distance: "+str(round(depth,3)),(200,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(124,252,0),2)
           cv2.putText(frame_right, "Distance:" + str(round(depth,3)), (200,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(124,252,0),2)
           print("Depth: ", depth)


       cv2.imshow("frame right", frame_right)
       cv2.imshow("frame left", frame_left)
       cv2.imshow("mask right", mask_right)
       cv2.imshow("mask left", mask_right)

       if cv2.waitKey(1) & 0xFF == ord('q'):
           break





cap_right.release()
cap_left.release()

cv2.destroyAllWindows()



           






