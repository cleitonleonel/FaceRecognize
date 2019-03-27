import numpy as np
import cv2
import os
from PIL import Image

#id = input('Entre com um usário')
dir = os.getcwd()
arq = dir + '/TrainedLabels.txt'
cascPath = dir + '/data/haarcascades/haarcascade_frontalface_default.xml'
#recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(dir + '/recognizer/trainningData.yml')


def getProfile(id):
    id_ = id[0]
    profile = None
    for line in open(arq,'r').readlines():
        if str(id_) == line.split(',')[0]:
            print('LINE:',line)
            profile = line.split(',')
            return profile

#id = 0
#font = (cv2.FONT_HERSHEY_SIMPLEX)
font = (cv2.FONT_HERSHEY_PLAIN)
#print (font)


def detect():
    faceDetect = cv2.CascadeClassifier(cascPath)
    cam = cv2.VideoCapture(0)
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray,1.3,5)
        #vis = img.copy()
        for(x,y,w,h) in faces:
            print('FACES:', faces)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            id = recognizer.predict(gray[y:y+h,x:x+w])
            id0 = id[0]
            id1 = id[1]
            print('SOU ID: ' + str(id))

            profile = getProfile(id)
            print('SOU PROFILE:',profile)

            p0 = profile[0]
            print('SOU P0: ' + str(p0))
            p1 = profile[1]
            print('SOU P1: ' + str(p1))
            p2 = profile[2]
            print('SOU P2: ' + str(p2))
            print(str(p0) + '==' + str(id0))
            if str(p0) == str(id0):
                cv2.putText(img, "Nome: "+str(profile[1]), (x, y + h + 30), font, 2, (0, 0, 255), lineType=cv2.LINE_AA);
                cv2.putText(img, "Idade: " + str(profile[2]), (x, y + h + 50), font, 1.0, (0, 0, 255), lineType=cv2.LINE_AA);
                #cv2.putText(img, "Genero: " + str(profile[3]), (x, y + h + 70), font, 0.4, (0, 0, 255), 1);
                #cv2.putText(img, "Crime: " +str(profile[4]), (x, y + h + 90), font, 0.4, (0, 0, 255), 1);
            elif not str(id0):
                cv2.putText(img, "Nao encontrado", (x, y + h + 30), font, 2, (0, 0, 255), 1)

        cv2.imshow('Face',img)
        if(cv2.waitKey(1) == ord('q')):
            print ('Parando...')
            break

    cam.release()
    cv2.destroyAllWindows()
