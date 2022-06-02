# EX1
# we will talk about missing data , header , dtype
import csv
import numpy as np
import pandas as pd 
FILE_NAME = "spambase.data"
data = np.genfromtxt(FILE_NAME,delimiter=",",dtype=np.float32,skip_header=1,missing_values="Hello"
                     ,filling_values=9999.0)
# data = np.loadtxt(FILE_NAME,delimiter=",")


with open(FILE_NAME,'r') as f:
    data = list(csv.reader(f,delimiter=","))
data = np.array(data)
print(data.shape,type(data[0][2]))
n_samples , n_features = data.shape
n_features -=1 
X = data[:,0:n_features]
y = data[:,n_features]
print(X.shape,y.shape)
print(X[0,0:5])

df = pd.read_csv(FILE_NAME,header=None,delimiter=",",dtype=np.float32,skiprows=1,na_values=["Uwazeer"])
df = df.fillna(9999.0)
data = df.to_numpy()
n_samples , n_features = data.shape
n_features -=1 
X = data[:,0:n_features]
y = data[:,n_features]
print(X.shape,y.shape)
print(X[0,0:5])


