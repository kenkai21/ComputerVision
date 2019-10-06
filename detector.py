import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer\\trainingData.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


cam = cv2.VideoCapture(0)
# font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.2,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im, (x,y), (x+w, y+h), (225, 0, 0), 2)
        Id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if (confidence > 99):
            if(Id==1):
                Id = 'Gozie'
            elif(Id==2):
                Id='James'
        else:
            Id="Unknown"
        # cv2.putText(im, (x, y+h), font, (225, 255, 255))
        # cv2.putText(im, (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), lineType=cv2.LINE_AA)
        # cv2.rectangle(im, str(Id), (x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(im, str(Id), (x, y+h), font, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow('im', im)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

