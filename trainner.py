import os
import cv2
import numpy as np
from PIL import Image
from detector import detect

#recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()
dir = os.getcwd()
path = dir + '/img/'


def getImages(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    print (imagePaths)
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert("L")
        print (faceImg)
        faceNp = np.array(faceImg,'uint8')
        print (faceNp)
        ID = int(os.path.split(imagePath)[-1].split('.')[0])
        print(ID)
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow('trainning',faceNp)
        cv2.waitKey(10)
    return IDs, faces


def trainner():
    Ids,faces = getImages(path)
    recognizer.train(faces,np.array(Ids))
    recognizer.write(dir + '/recognizer/trainningData.yml')
    cv2.destroyAllWindows()
    detect()
