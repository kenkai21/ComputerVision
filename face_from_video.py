import numpy as np
import cv2, time


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    # Our operations on the frame come here
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features =classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x,y,w,h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]

    return coords


def detect(img, faceCascade, eyesCascade):
    color = {'blue':(255,0,0), 'green':(0,255,0)}
    coords = draw_boundary(img, faceCascade, 1.1, 5, color['green'], "Face")

    if len(coords)==4:
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[3]]
        coords = draw_boundary(roi_img, eyesCascade, 1.1, 5, color['blue'], "Eyes")
    return img


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyesCascade = cv2.CascadeClassifier('haarcascade_eye.xml')


cap = cv2.VideoCapture(0)
while(True):
   
    # Capture frame-by-frame
    _, img = cap.read()
    img = detect(img, faceCascade, eyesCascade)

    # # Display the resulting frame
    cv2.imshow('face detection', img)
  
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()