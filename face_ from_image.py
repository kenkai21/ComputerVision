import cv2
import numpy as np 


face_csc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


img = cv2.imread('richfacepoorface.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faces = face_csc.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

resized_image = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))

cv2.imshow('stock_people', resized_image)

cv2.waitKey(0)


cv2.destroyAllWindows()

# print(type(faces))
# print(faces)