from sklearn.svm import SVC
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("D:\SBE\Third Year\Second Term\Computer Vision\Tasks\Task5\Face_Recognition_Lab\data\pca_features.csv")
y = df.iloc[:, 0].values       # the label that represent the person
X = df.iloc[:, 1:].values     # the others is features

"""just testing and tha accuracy is 0.97"""
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# clf = SVC(kernel='rbf', probability=True) # allow prob to estimate whether it significant or not to reject person outside data
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print("Accuracy:", acc)
"""train on whole data"""
clf = SVC(kernel='rbf', probability=True)
clf.fit(X, y)



# save weight
joblib.dump(clf, "svm_pca_model.pkl")
print("done y basha")