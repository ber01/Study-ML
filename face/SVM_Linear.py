from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import metrics

faces = datasets.fetch_olivetti_faces()
X = faces.data
y = faces.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(metrics.classification_report(y_test, y_pred))
