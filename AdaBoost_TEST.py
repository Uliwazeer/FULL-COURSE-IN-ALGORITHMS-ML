from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from adaboost import Adaboost
#data = datasets.load_digits()


data = datasets.load_breast_cancer()
X = data.data
y = data.target

# project the data onto the two library principal components
y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=5)

# Adaboost classification with 5 weaks classifiers
clf = Adaboost(n_clf=5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print("Accuracy:", acc)
