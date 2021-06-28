# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 18:57:40 2021

@author: ikrilis
"""

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Dataset
datapath = 'data/'
dataset = pd.read_csv(datapath + 'train.csv')
features = dataset.copy()

# Date Features
features['date'] = pd.to_datetime(dataset['date'])
features['year'] = features.date.dt.year
features['month'] = features.date.dt.month
features['day'] = features.date.dt.day
features['dayofyear'] = features.date.dt.dayofyear
features['dayofweek'] = features.date.dt.dayofweek
features['weekofyear'] = features.date.dt.isocalendar().week

# Additionnal Data Features
features['daily_avg']  = features.groupby(['item','store','dayofweek'])['sales'].transform('mean')
features['weeky_avg']  = features.groupby(['item','store','weekofyear'])['sales'].transform('mean')
rolling_10 = features.groupby(['item'])['sales'].rolling(10).mean().reset_index().drop('level_1', axis=1)
features['rolling_mean'] = rolling_10['sales'].copy()

# Drop date
features.drop('date', axis=1, inplace=True)
features = features.dropna()

#%% Feature Analysis
# Correlation with the target value
plt.figure()
corr = features.drop(['store', 'item'], axis=1).corr()
sn.heatmap(corr, annot = True)

plt.figure()
series = dataset[["date", "sales"]]
plt.plot(features['weeky_avg'])


#%% Train Test Split
X_train , X_test ,y_train, y_test = train_test_split(
    features.drop('sales',axis=1),
    features.pop('sales'), 
    test_size=0.2,
    shuffle = False,
    stratify = None)

scaler = StandardScaler().fit(X_train)
X_train= scaler.transform(X_train)
X_test = scaler.transform(X_test)

np.save(datapath + "X_train.npy", X_train)
np.save(datapath + "X_test.npy", X_test)
np.save(datapath + "y_train.npy", y_train)
np.save(datapath + "y_test.npy", y_test)
