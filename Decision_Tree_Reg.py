#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 19:59:53 2019

@author: zaidbhat
"""


#Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing Dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:, 2].values
#Splitting the data into traning and test set
#from sklearn.model_selection import train_test_split
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0) 

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

 
#Fitting DT Regression Model
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)


#Predicting new result using DTregression
y_pred = regressor.predict([[6.5]])



#Visualising Regression results for High Resolution and smoother curve
X_grid = np.arange(min(X),max(X),0.1)
X_grid = np.reshape((len(X_grid),1))
plt.scatter(X, Y,color='red')
plt.plot(X_grid, regressor.predict(X_grid),color='blue')
plt.title('Truth or Bluff( Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()