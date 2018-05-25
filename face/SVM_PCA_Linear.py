from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn import decomposition

faces = datasets.fetch_olivetti_faces()
X = faces.data
y = faces.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

pca = decomposition.PCA(n_components=150, whiten=True)
pca.fit(X_train)

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

clf = svm.SVC(kernel='linear')
clf.fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)

print(metrics.classification_report(y_test, y_pred))
