import pickle
import numpy as np
import os
import cv2
import time
from os import listdir
from PIL import Image as Img
from keras_facenet import FaceNet
facemodel = FaceNet()
#read file vector
# myfile = open('new_data.pkl','rb')
# data = pickle.load(myfile)
# myfile.close()
Labels=[]
Vector=[]
# for names, vector in data.items():
#     Labels.append(names)
#     Vector.append(vector)
#

path = './dataset/'

for filename in os.listdir(path):
    for items in os.listdir(path+filename+"/"):
        # print(items)
        # print(str(filename))
        img = cv2.imread(path+filename+"/"+items)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(160,160))
        img= img.reshape(1,160,160,3)
        signature = facemodel.embeddings(img)
        Labels.append(str(filename))
        Vector.append(signature)
Labels = np.asarray(Labels)
Vector = np.asarray(Vector)
print(Labels)
print(Vector)
with open('LV.npy','wb') as f:
    np.save(f,Labels)
    np.save(f,Vector)
