
# coding: utf-8

# In[1]:

from sklearn import datasets
from sklearn.model_selection import cross_val_predict, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import metrics
from sklearn import decomposition
import numpy as np
faces = datasets.fetch_olivetti_faces()
X = faces.data
y = faces.target

clf = svm.SVC(C=5., gamma=0.001)

kf = KFold(n_splits=10, shuffle=True, random_state=0)

y_pred = list()
y_test_result = list()
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    clf.fit(X_train, y_train)
    y_pred.extend(clf.predict(X_test))
    y_test_result.extend(y_test)

print(metrics.classification_report(y_test_result, y_pred))


# In[ ]:



