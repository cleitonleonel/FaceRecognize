#!/usr/bin/env python
# coding: utf-8

import numpy as np
from trainner import trainner
import glob
import cv2
import os

sampleNum = 0
dir = os.getcwd()
print(dir)
cascPath = dir + '/data/haarcascades/haarcascade_frontalface_default.xml'
faceDetect = cv2.CascadeClassifier(cascPath)
check_user = glob.glob(dir + "/img" + "/*.jpg")
print('CHECK USER:', check_user)


def check():
    print('CHEGUEI AQUI...')
    if check_user is not []:
        print('Deseja cadastrar novo usuário?')
        text = input('')
        if 'sim' in text or 'Sim' in text or 'SIM' in text:
            id = input('Entre com um id: ')
            user = input('Entre com um usuário: ')
            idade = input('Digite sua idade: ')
            ficheiro = dir + "/TrainedLabels.txt"
            arquivo = open(ficheiro, 'a')
            arquivo.write(str(id) + ',' + str(user) + ',' + str(idade) + ',' + '\n')
            arquivo.close()
            capture(id)
        else:
            trainner()
    return True


def capture(id=None):
    cam = cv2.VideoCapture(0)
    sampleNum = 0
    while (True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            sampleNum += 1
            cv2.imwrite(dir + "/img/" + str(id) + "." + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.waitKey(100)
        cv2.imshow('Face', img)
        cv2.waitKey(1)
        if (sampleNum > 50):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start = check()
