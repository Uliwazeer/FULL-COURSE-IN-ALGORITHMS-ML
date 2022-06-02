from PERCEPTRON_TEST import X_test, X_train
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn import datasets 
import matplotlib.pyplot as plt 
from random_forest import RandomForest




def accuracy(y_pred,y_true):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

data = datasets.load_breast_cancer()
X = data.data 
y = data.target 


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)
clf = RandomForest(n_trees=5)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
acc = accuracy(y_test,y_pred)

print("Accuracy:", acc)