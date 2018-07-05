# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 20:10:04 2018

@author: Anirudh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values.reshape(dataset.iloc[:,-1].values.shape[0],1)

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# first Encode the string / categorical variable 
X[:,-1] = LabelEncoder().fit_transform(X[:,-1])

ohe_X = OneHotEncoder(categorical_features = [3])
X = ohe_X.fit(X).transform(X).toarray()
# avoid dummy variable trap
X = X[:,1:]

from sklearn.preprocessing import StandardScaler
std_scaler_X = StandardScaler()
std_scaler_X.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
fit_X_train = regressor.fit(X_train, y_train)

y_test_predictions = fit_X_train.predict(X_test)


# Backward Elimination for building the optimal model with 
# elimination criterion p-values > 0.5
import statsmodels.formula.api as sm

# the following step is to introduce X_0 for which coefficient will be b_0
# because the statsmodels library doesn't implicitly account for it
X = np.append(arr = np.ones(shape = (X.shape[0],1)), values = X, axis = 1)

X_opt = X[:,np.arange(0,X.shape[1])]
regressor_OLS = sm.OLS(endog = y, exog = X_opt)
fit_X = regressor_OLS.fit()
print(fit_X.summary())

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt)
fit_X = regressor_OLS.fit()
print(fit_X.summary())

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt)
fit_X = regressor_OLS.fit()
print(fit_X.summary())

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt)
fit_X = regressor_OLS.fit()
print(fit_X.summary())

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt)
fit_X = regressor_OLS.fit()
print(fit_X.summary())





