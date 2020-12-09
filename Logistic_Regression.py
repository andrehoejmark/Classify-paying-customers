import numpy as np
import math
import time
import copy


'''
Classifies an input to the probability of being classified into a class.

'''
class Logistic_Regression:


    def __init__(self, lr, n_iterations):
        self.lr = lr
        self.n_iterations = n_iterations
    
    def fit(self, X, y):
        
        # setup weights/thetas
        n_samples, n_features = X.shape
        self.W = np.array([0] * n_features)
        
        # update weights
        for _ in range(self.n_iterations):
            
            z = np.dot(X, self.W)
            y_hat = self._sigmoid(z)
            gradient = (1/n_samples) * np.dot(X.T, (y_hat-y))
            self.W = self.W - self.lr * gradient
            
                   
    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def predict(self, X):
        
        predictions = []
        for row in X.values.astype(np.float):
            
            proba = self._sigmoid(np.dot(self.W, row))
            
            if proba >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)
            
        return predictions 

def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc

























     
    