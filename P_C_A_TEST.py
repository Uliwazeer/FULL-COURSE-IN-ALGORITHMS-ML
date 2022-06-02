from sklearn import datasets 
import matplotlib.pyplot as plt 
import numpy as np 
from pca import PCA 
#data = datasets.load_digits()


data = datasets.load_iris()
X = data.data 
y = data.target 

# project the data onto the two library principal components
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print('Shape of X:', X.Shape)
print('Shape of transformed:', X_projected.Shape)


x1 = X_projected[:,0]
x2 = X_projected[:,1]

plt.scatter(x1,x2,
            c=y,edgecolor='none',alpha=0.8,cmap=plt.cm.get_cmap('viridis',3))


plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.colorbar()
plt.show()