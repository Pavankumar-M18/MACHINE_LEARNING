# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 12:50:23 2024

@author: pk161
"""

# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv(r"C:\Users\pk161\OneDrive\DATA\Investment.csv")

# indipendent variable
x = dataset.iloc[:,:-1]

# dependent variable
y = dataset.iloc[:,4]

# converting category into numbers in x
x = pd.get_dummies(x,dtype=int)

# split the dataset
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

# creating the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# fitting the model
regressor.fit(x_train,y_train)

# predict the model
y_pred = regressor.predict(x_test)

# extracting the coefficient
m = regressor.coef_
print(m)

# extract the intercept
c = regressor.intercept_
print(c)

x = np.append(arr = np.ones((50,1)).astype(int),values=x,axis=1)


import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,4,5]]

# ordanary least squares
# creating OLS model
regressor_ols = sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()


import statsmodels.api as sm
x_opt = x[:,[0,1,2,3,4]]

# OLS
regressor_ols = sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()



import statsmodels.api as sm
x_opt = x[:,[0,1,2,3]]

# craete ols model
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
regressor_ols.summary()


import statsmodels.api as sm
x_opt = x[:,[0,1,2]]

# ols model
regressor_ols = sm.OLS(endog = y,exog = x_opt).fit()
regressor_ols.summary()



import statsmodels.api as sm
x_opt = x[:,[0,1]]

regressor_ols = sm.OLS(endog=y,exog=x_opt).fit()
regressor_ols.summary()


bias = regressor.score(x_train,y_train)
bias

variance = regressor.score(x_test,y_test)
variance
