import cv2
import os
import numpy as np
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
cam = cv2.VideoCapture(0) #commencer video
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Detecter un objet dans un flux
id=raw_input('enter user id')
count = 0
assure_path_exists("criminel/")
while(True):
    _, img = cam.read() #capture image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #Convertir un cadre en niveaux de gris
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)#Recadrer le cadre de l'image en rectangle
        count += 1
        cv2.imwrite("criminel/User." + str(id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.waitKey(100)
    cv2.imshow('frame', img)
    cv2.waitKey(1)
    if (count>20):
        break
cam.release()
cv2.destroyAllWindows()
