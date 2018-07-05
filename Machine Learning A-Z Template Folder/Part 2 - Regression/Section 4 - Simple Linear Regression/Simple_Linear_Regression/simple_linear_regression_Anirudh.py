# -*- coding: utf-8 -*-
"""
Created on Wed May 30 09:35:31 2018

@author: Anirudh
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# list files in the current directory

dataset = pd.read_csv('Salary_Data.csv')

from sklearn.model_selection import train_test_split


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values.reshape(dataset.iloc[:,-1].values.shape[0],1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3,
                                                    random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressedModel = regressor.fit(X_train, y_train)
y_test_preds = regressedModel.predict(X_test)

plt.scatter(X_train,y_train,c = 'b')
plt.plot(X_train, regressedModel.predict(X_train), c = 'r')
plt.show()

