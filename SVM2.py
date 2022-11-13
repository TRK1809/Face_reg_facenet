import numpy as np
import sklearn.metrics
from sklearn import svm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sn
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

with open('LV.npy','rb') as f:
    y = np.load(f) #  iD #

    X = np.load(f) # vector
#print(X)
#print(y)
X=np.squeeze(X,axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#print(y_test[0])
clf = svm.SVC()
clf.fit(X_train,y_train)

preds = clf.predict(X_test)
val_acc = accuracy_score(y_test, preds)
#print(val_acc)
filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
test_vector = np.expand_dims(X_test[0],axis = 1)

result = loaded_model.predict(X_test)
print(val_acc)
print(precision_recall_fscore_support(y_test,preds,average="macro"))
cm = confusion_matrix(y_test, preds, labels=clf.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot(xticks_rotation="vertical")
plt.show()


