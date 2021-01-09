import cv2
import numpy as np

#using cascade classifier
smile_cascade = cv2.CascadeClassifier("smile.xml")
#procedure to start video capture using primary camera
cap = cv2.VideoCapture(0)

#condition if camera is on
while True:
    #reading frames 
    ret, img = cap.read()
    #converting frames to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #multiscale to avoid multiple roi
    smile = smile_cascade.detectMultiScale(gray, 1.3, 5)
    #for detecting the co-ordinates of found roi
    for (x,y,w,h) in smile:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        smile1 = smile_cascade.detectMultiScale(roi_gray,
                                               scaleFactor=1.7,
                                               minNeighbors=22,
                                               minSize=(25, 25),
                                               )
        #for laying the co-ordinates of found roi
        for (ex, ey, ew, eh) in smile1:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
    
    #display contents of continuous frames with the ROI   
    cv2.imshow('img', img)
    #method to exit the window
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
        
    