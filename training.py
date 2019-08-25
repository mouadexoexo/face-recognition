
import cv2, os
import numpy as np
from PIL import Image
import os
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
recognizer = cv2.face.LBPHFaceRecognizer_create()#Creation d'histogrammes pour la reconnaissance des visages
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");#Detecter un objet dans un flux
def getImagesAndLabels(path):#obtenir les images et les donnees d etiquette
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]#Obtenir tout le chemin du fichier
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')#convertir en PIL image
        img_numpy = np.array(PIL_img,'uint8')#convertir en numpy array
        id = int(os.path.split(imagePath)[-1].split(".")[1])#obtenir image id
        faces = detector.detectMultiScale(img_numpy)#Obtenir le visage des images de formation
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
faces,ids = getImagesAndLabels('criminel')
recognizer.train(faces, np.array(ids))#Former le modele a laide des faces et des identifiants
assure_path_exists('trainer/')
recognizer.save('trainer/trainer.yml')

