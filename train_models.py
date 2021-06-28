# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 19:31:45 2021

@author: ikrilis
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid


#%% Load data
datapath = 'data/'
X_train = np.load(datapath + "X_train.npy")
X_test = np.load(datapath + "X_test.npy")
y_train = np.load(datapath + "y_train.npy")
y_test = np.load(datapath + "y_test.npy")
dataset = pd.read_csv(datapath + 'train.csv')
dates_train = dataset['date'].iloc[:X_train.shape[0]]
dates_test = dataset['date'].iloc[X_train.shape[0]:]
feat_names = ['store', 'item', 'year', 'month', 'day', 'dayofyear', 'dayofweek',
       'weekofyear', 'daily_avg', 'weeky_avg', 'rolling_mean']

# Shuffle train data
ind = np.arange(X_train.shape[0])
np.random.shuffle(ind)
X_train = X_train[ind, :]
y_train = y_train[ind]

#%% Linear Regression
model_1 = LinearRegression()
model_1.fit(X_train, y_train)
yhat_train_1 = model_1.predict(X_train)
yhat_test_1 = model_1.predict(X_test)

train_err_1 = mean_squared_error(y_train, yhat_train_1)
test_err_1 = mean_squared_error(y_test, yhat_test_1)
forecast_error_1 = y_test - yhat_test_1
print('Min Forecast error', min(forecast_error_1))
print('Max Forecast error', max(forecast_error_1))
print('Abs Mean Forecast error', np.mean(np.abs(forecast_error_1)))

# Calc importances
feat_imp_1 = np.abs(model_1.coef_)
indices_1 = np.argsort(feat_imp_1)[::-1]

#%% XGBoost
# XGB Model
matrix_train = xgb.DMatrix(X_train, label = y_train)
matrix_test = xgb.DMatrix(X_test, label = y_test)

# Run XGB 
model_2 = xgb.train(params={'objective':'reg:squarederror','eval_metric':'rmse',
                            },
                            early_stopping_rounds = 20,
                            dtrain = matrix_train, num_boost_round = 100,
                            evals = [(matrix_test,'test')],)

yhat_train_2 = model_2.predict(matrix_train)
yhat_test_2 = model_2.predict(matrix_test)

train_err_2 = mean_squared_error(y_train, yhat_train_2)
test_err_2 = mean_squared_error(y_test, yhat_test_2)
forecast_error_2 = y_test - yhat_test_2
print('Min Forecast error', min(forecast_error_2))
print('Max Forecast error', max(forecast_error_2))
print('Abs Mean Forecast error', np.mean(np.abs(forecast_error_2)))

# Calc importance
imp_dict = model_2.get_score(importance_type='gain')
indices_2 = [int(k[1:]) for k in imp_dict.keys()] 
feat_imp_2 = [v for v in imp_dict.values()] 


#%% XGB Model Tuned with Cross validation
matrix_train = xgb.DMatrix(X_train, label = y_train)
matrix_test = xgb.DMatrix(X_test, label = y_test)

res_cv = []
params = {'objective':['reg:squarederror'],'eval_metric':['rmse'],'eta':[0.2],
          'max_depth':[3, 6, 9], 'subsample':[0.8, 0.9], 'lambda':[1, 10] 
          }

# Run XGB 
param_grid = list(ParameterGrid(params))

for i in range(len(param_grid)):
    print("Current parameter id:", i)
    res_cv.append( xgb.cv(params=param_grid[i],
                          dtrain = matrix_train, num_boost_round = 500,
                          early_stopping_rounds=50, nfold = 3
                          ))

val_rmse = [res['test-rmse-mean'].values[-1] for res in res_cv]
best_iter = np.argmin(val_rmse)
best_params = param_grid[best_iter]

# Run XGB 
model_cv = xgb.train(params=best_params,
                        early_stopping_rounds = 50,
                        dtrain = matrix_train, num_boost_round = 500,
                        evals = [(matrix_test,'test')],)

yhat_train_cv = model_cv.predict(matrix_train)
yhat_test_cv = model_cv.predict(matrix_test)

train_err_cv = mean_squared_error(y_train, yhat_train_cv)
test_err_cv = mean_squared_error(y_test, yhat_test_cv)
forecast_error_cv = y_test - yhat_test_cv
print('Min Forecast error', min(forecast_error_cv))
print('Max Forecast error', max(forecast_error_cv))
print('Abs Mean Forecast error', np.mean(np.abs(forecast_error_cv)))

# Calc importance
imp_dict = model_cv.get_score(importance_type='gain')
indices_cv = [int(k[1:]) for k in imp_dict.keys()] 
feat_imp_cv = [v for v in imp_dict.values()] 
#%% Plots
# Forecasting of the two models
n_days = 90 # ~3months
plt.figure()
plt.plot(dates_test[:n_days], y_test[:n_days], '-.k')
plt.plot(dates_test[:n_days], yhat_test_1[:n_days])
plt.plot(dates_test[:n_days], yhat_test_cv[:n_days], 'r')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.gcf().autofmt_xdate()
plt.grid(linewidth=0.6)
plt.legend(["Ground Truth", "Linear Regression", "XGBoost"])
# plt.title("Sales Forecasting")
plt.title("Sales (3 months)")
plt.xlabel('Date')
plt.ylabel('Sales')

# LR coefficients
plt.figure()
plt.barh(np.arange(len(indices_1)), feat_imp_1[indices_1])
feat_names_sorted = [feat_names[i] for i in indices_1 ]
plt.yticks(np.arange(len(indices_1)), feat_names_sorted)
plt.xlabel('Feature Importance (Linear Regression)')

# Xgboost gain
plt.figure()
plt.barh(np.arange(len(indices_2)), feat_imp_2)
feat_names_sorted = [feat_names[i] for i in indices_2 ]
plt.yticks(np.arange(len(indices_2)), feat_names_sorted)
plt.xlabel('Feature Importance (XGBoost)')

#%% LSTM
# def create_inout_sequences(input_data, tw):
#     inout_seq = []
#     L = len(input_data)
#     for i in range(L-tw):
#         train_seq = input_data[i:i+tw]
#         train_label = input_data[i+tw:i+tw+1]
#         inout_seq.append((train_seq ,train_label))
#     return inout_seq


# class LSTM(nn.Module):
#     def __init__(self, input_size=1, hidden_layer_size=10, output_size=1):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size

#         self.lstm = nn.LSTM(input_size, hidden_layer_size)

#         self.linear = nn.Linear(hidden_layer_size, output_size)

#         self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
#                             torch.zeros(1,1,self.hidden_layer_size))

#     def forward(self, input_seq):
#         lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
#         predictions = self.linear(lstm_out.view(len(input_seq), -1))
#         return predictions[-1]

# train_data = torch.FloatTensor(y_train).view(-1)
# train_window = 10
# train_inout_seq = create_inout_sequences(train_data, train_window)
# model = LSTM()
# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# epochs = 2

# for i in range(epochs):
#     for seq, labels in train_inout_seq:
#         optimizer.zero_grad()
#         model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
#                         torch.zeros(1, 1, model.hidden_layer_size))

#         y_pred = model(seq)

#         single_loss = loss_function(y_pred, labels)
#         single_loss.backward()
#         optimizer.step()

#     # if i%1 == 1:
#         print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

# print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
