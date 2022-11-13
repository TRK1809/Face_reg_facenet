import numpy as np
from numpy import asarray
import cv2
import os
from os import listdir
import pickle
from PIL import Image as Img
from keras_facenet import FaceNet
Facemodel = FaceNet()

database={}
folder = './Test/'
for filename in listdir(folder):

    path = folder + filename
    gbr1 = cv2.imread(folder + filename)
    gbr = cv2.cvtColor(gbr1, cv2.COLOR_BGR2RGB)
    gbr = Img.fromarray(gbr)
    face = asarray(gbr)
    face = asarray(face)
    face = face.astype('float32')
    face = cv2.resize(face,(160,160))
    face = face.reshape(1,160,160,3)
    signature = Facemodel.embeddings(face)

    database[os.path.splitext(filename)[0]]=signature
#print(database)
myfile = open('new_data.pkl', 'wb')

pickle.dump(database, myfile)