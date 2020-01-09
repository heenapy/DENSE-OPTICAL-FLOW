#____________________DENSE OPTICAL FLOW______________________________________

import numpy as np
import cv2
cap = cv2.VideoCapture("/home/paython/Videos/4K Video Downloader/Walking through Japan in 1 min.mp4")

ret, first_frame = cap.read()
previous_gray = cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(first_frame)
hsv[...,1]=255

while True:
    ret, frame = cap.read()
    next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    flow  = cv2.calcOpticalFlowFarneback(previous_gray,next,None,0.5,3,15,3,5,1.2,0)
    magnitude, angle = cv2.cartToPolar(flow[...,0],flow[...,1])
    hsv[...,0] = angle * (180/(np.pi/2))
    hsv[...,2] = cv2.normalize(magnitude,None,0,255,cv2.NORM_MINMAX)
    final = cv2.cvtColor(hsv,cv2.COLOR_HLS2BGR)
    cv2.imshow('Dense optical flow',final)
    if cv2.waitKey(1)==13:
        break
    previous_gray=next

cv2.destroyAllWindows()
cap.release()