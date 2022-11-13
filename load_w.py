import keras_facenet
import cv2
import matplotlib.pyplot as plt
import pickle
import PIL
from PIL import Image as Img
from keras_facenet import FaceNet
import os
import numpy as np
from numpy import  asarray
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
faceCascade = cv2.CascadeClassifier(cascPath)
embedder = FaceNet()
img = cv2.imread("Test/cr7.jpg")
imgcopy = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face = faceCascade.detectMultiScale(gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
for (x,y,w,h) in face:
    coords = [x, y, w, h]
    cv2.rectangle(imgcopy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    crop = imgcopy[coords[1]: coords[1] + coords[3], coords[0]: coords[0] + coords[2]]
    crop = Img.fromarray(crop)
    crop = asarray(crop)
    crop = crop.astype('float32')
    crop = cv2.resize(crop, (160, 160))
    crop = crop.reshape(1, 160, 160, 3)
    signature = embedder.embeddings(crop)
    identity = str(loaded_model.predict(signature)[0])
    cv2.putText(imgcopy, identity, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0,), 1, cv2.LINE_AA)
cv2.imshow("{}".format(identity),imgcopy)
cv2.imwrite("cr7_detected.jpg",imgcopy)
cv2.waitKey()



