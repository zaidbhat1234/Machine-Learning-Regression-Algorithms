#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:24:30 2019

@author: zaidbhat
"""

#SVR


#Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing Dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:, 2].values
Y=  Y.reshape(-1, 1)
#Splitting the data into traning and test set
#from sklearn.model_selection import train_test_split
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0) 

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

 
#Fitting SVR Model
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,Y)


#Predicting new result using Polynomial regression
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]])))
y_pred = sc_Y.inverse_transform(y_pred)

#Visualising SVR results
plt.scatter(X, Y,color='red')
plt.plot(X, regressor.predict(X),color='blue')
plt.title('Truth or Bluff( SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()