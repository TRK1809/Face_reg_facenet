import keras_facenet
import numpy as np
from numpy import asarray
import cv2
import os
import pickle
import PIL
from PIL import Image as Img
from keras_facenet import FaceNet

#read file vector
myfile = open('new_data.pkl','rb')
data = pickle.load(myfile)
myfile.close()
#print(data)
#load_model

embedder = FaceNet()
print(type(FaceNet()))
Test_img = cv2.imread('dataset/Hien/2_2.jpg')
Test_img = cv2.cvtColor(Test_img, cv2.COLOR_BGR2RGB)
Test_img = Img.fromarray(Test_img)
Test_img = asarray(Test_img)
face = asarray(Test_img)
face = face.astype('float32')
# mean, std = face.mean(), face.std()
# face = (face - mean) / std
face = cv2.resize(face, (160, 160))
face = face.reshape(1, 160, 160, 3)
signature = embedder.embeddings(face)

min_dist =10
smol = []
identity =''
for key, value in data.items():
    khoang_cach = value-signature
    khoang_cach =np.linalg.norm(khoang_cach)
    smol.append(khoang_cach)
    if khoang_cach < min_dist:
        min_dist = khoang_cach
        identity = key
print(identity)
print(min_dist)
print(smol)
print(signature)