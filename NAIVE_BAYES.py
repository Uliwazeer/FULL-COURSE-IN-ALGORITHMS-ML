# naive bayes in python for machine learning from scratch
# EX1
from types import new_class
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


class NaiveBayes:
    def fit(self,X,y):
      n_samples,n_features = X.shape 
      self._classes = np.unique(y)
      new_classes = len(self._classes)
      
      
        #init mean var priors
        self._mean = np.zeros((new_classes,n_features),dtype=np.float64) 
        self._var = np.zeros((new_classes,n_features),dtype=np.float64) 
        self._priors = np.zeros(new_classes,dtype=np.float64) 
        
        
        for c in self._classes:
            X_c =X[c==y]
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)
        
        
    def predict(self,X,y):
        y_pred = [self._predict(x) for x in X]
        return y_pred
    
    def _predict(self,x):
        posteriors = []
        for idx , c in enumerate(self._classes):
            prior = np.log(self._priors[idx]) 
            class_conditional = np.sum(np.log(self._pdf(idx,x)))
            posteriors.append(posteriors)
            
            return self._classes[np.argmax(posteriors)]
        
    def _pdf(self,class_idx,x):
        mean = self._mean(class_idx)    
        var = self._var(class_idx)    
        
        numerator = np.exp(-(x-mean)**2/(2*var))
        denominator = np.sqrt(2*np.pi * var)
        return numerator / denominator
    