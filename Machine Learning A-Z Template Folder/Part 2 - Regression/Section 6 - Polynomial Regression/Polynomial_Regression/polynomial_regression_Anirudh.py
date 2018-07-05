# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 13:21:33 2018

@author: Anirudh
"""

import os
PATH = 'F:\\ML_Udemy\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 6 - Polynomial Regression\\Polynomial_Regression'
os.chdir(PATH)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,:-1].values
# but then we don't need the first column of X
X = X[:,1].reshape(X.shape[0],1)
y = dataset.iloc[:,-1].values.reshape(dataset.iloc[:,-1].shape[0],1)

# First try out linear model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# linear model based predictions:
linear_preds_y = LinearRegression().fit(X,y).predict(y)

from sklearn.preprocessing import PolynomialFeatures
poly_X = PolynomialFeatures(degree = 3).fit_transform(X)

# polynomial model based predictions:
poly_preds_y = LinearRegression().fit(poly_X,y).predict(poly_X)


# Visualizing Linear Regression
plt.plot(X,linear_preds_y, c = 'b')
# plt.show()
plt.scatter(X,y,c = 'r')
plt.show()

# Visualizing Polynomial Regression
plt.plot(poly_X[:,1],poly_preds_y, c = 'b')
plt.scatter(X,y,c = 'r')
plt.show()


# For a more even graph 

X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape(X_grid.shape[0],1)
poly_X_grid = PolynomialFeatures(degree = 3).fit_transform(X_grid)
poly_preds_y_X_grid = LinearRegression().fit(poly_X,y).predict(poly_X_grid)

plt.plot(poly_X_grid[:,1],poly_preds_y_X_grid, c = 'b')
plt.scatter(X,y,c = 'r')
plt.show()



