# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 19:31:45 2021

@author: ikrilis
"""

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#%% Load data
datapath = 'data/'
X_train = np.load(datapath + "X_train.npy")
X_test = np.load(datapath + "X_test.npy")
y_train = np.load(datapath + "y_train.npy")
y_test = np.load(datapath + "y_test.npy")


#%% Linear Regression
model_1 = LinearRegression()
model_1.fit(X_train, y_train)
yhat_train_1 = model_1.predict(X_train)
yhat_test_1 = model_1.predict(X_test)

train_err_1 = mean_squared_error(y_train, yhat_train_1)
test_err_1 = mean_squared_error(y_test, yhat_test_1)

plt.figure()
plt.plot(y_train)
plt.plot(yhat_train_1)
plt.figure()
plt.plot(y_test)
plt.plot(yhat_test_1)

#%% XGBoost
# XGB Model
matrix_train = xgb.DMatrix(X_train, label = y_train)
matrix_test = xgb.DMatrix(X_test, label = y_test)

# Run XGB 
model_2 = xgb.train(params={'objective':'reg:linear','eval_metric':'mae'}
                ,dtrain = matrix_train, num_boost_round = 500, 
                early_stopping_rounds = 20, evals = [(matrix_test,'test')],)

yhat_train_2 = model_2.predict(matrix_train)
yhat_test_2 = model_2.predict(matrix_test)

train_err_2 = mean_squared_error(y_train, yhat_train_2)
test_err_2 = mean_squared_error(y_test, yhat_test_2)

plt.figure()
plt.plot(y_train)
plt.plot(yhat_train_2)
plt.figure()
plt.plot(y_test)
plt.plot(yhat_test_2)


#%% LSTM

