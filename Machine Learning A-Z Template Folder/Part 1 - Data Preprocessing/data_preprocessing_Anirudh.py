# -*- coding: utf-8 -*-
"""
Created on Sun May 20 13:15:54 2018

@author: Anirudh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

# Imputing Missing Values with Mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:3]) # in order to fit the imputer object to X
X[:,1:3] = imputer.transform(X[:,1:3])

# Encode Categorical Variables
from sklearn.preprocessing import LabelEncoder
le_X = LabelEncoder()
le_Y = LabelEncoder()
le_X.fit(X[:,0]) # don't expect this to do much
X[:,0] = le_X.fit_transform(X[:,0])
y = le_Y.fit_transform(y)

from sklearn.preprocessing import OneHotEncoder
ohe_X = OneHotEncoder(categorical_features = [0]) # array of indices required
X = ohe_X.fit_transform(X).toarray()

# Split data into Train and Test Sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8,
                                                    random_state = 42)

# Scale the data - the same scaler is used for both train and test sets
from sklearn.preprocessing import StandardScaler
stdScaler_X = StandardScaler()
X_train = stdScaler_X.fit_transform(X_train)
X_test = stdScaler_X.transform(X_test)




