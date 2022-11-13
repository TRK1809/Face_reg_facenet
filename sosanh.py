from keras_facenet import FaceNet
import numpy as np
import cv2
import time
import pickle
import PIL.Image as Img
from numpy import asarray
embedder= FaceNet()
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
#read file vector
myfile = open('new_data.pkl','rb')
data = pickle.load(myfile)
myfile.close()
img = cv2.imread('dataset/Khang/1_8.jpg')
crop = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
crop = Img.fromarray(crop)
crop = asarray(crop)
crop = crop.astype('float32')
crop = cv2.resize(crop, (160, 160))
crop = crop.reshape(1, 160, 160, 3)
signature = embedder.embeddings(crop)
print(signature)
#time using SVM
atime = time.time()
identity = str(loaded_model.predict(signature)[0])
ctime = float(time.time()-atime)
print("excecute time svm = ",ctime)
print(identity)
#time using euclide
btime = time.time()

min_dist = 10
id = ''
for key, value in data.items():
    dist = np.linalg.norm(value - signature)
    if dist < min_dist:
        min_dist = dist
        id = key
dtime= float(time.time() - btime)
print("excecute time euclide = ",dtime)

print((dtime<ctime))


