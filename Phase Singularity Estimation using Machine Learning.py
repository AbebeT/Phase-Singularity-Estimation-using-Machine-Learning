# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:03:29 2019

@author: User
"""

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import preprocessing
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from keras.models import model_from_json


def get_data():
    dataset_path = 'dataset.csv'
    dataset = pd.read_csv(dataset_path)
    data = pd.read_csv(dataset_path, usecols=['FE','SD_FE','FD', 'SD_FD'])
    label = dataset['Average_PS']
    label=np.array(label)
    data = np.array(data)    
    data, label = shuffle(data,label)   
    data, label = shuffle(data,label)   
    data_min_max_scaler = MinMaxScaler(feature_range = (0,1))
    data = data_min_max_scaler.fit_transform(data)
    
    label = label.reshape(-1,1)
    label_scaler = MinMaxScaler(feature_range = (0,1))
    label = label_scaler.fit_transform(label)
    return data, label
 
def create_model():
     
    #create model
    my_model = Sequential()
    
    #Input Layer
    my_model.add(Dense(4,kernel_initializer='normal',input_dim = 4, activation = 'relu'))
    
    # The hidden Layers:
    my_model.add(Dense(3, kernel_initializer='normal', activation ='relu'))
    #my_model.add(Dropout(0.2))
    my_model.add(Dense(1, kernel_initializer='normal', activation = 'linear'))
    # Output Layer:
    my_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae'])
    #my_model.summary()
    return my_model


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
data, label = get_data()

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42)
# Model Training
model = KerasRegressor(build_fn=create_model, epochs=350, batch_size=10, verbose=0)

kf = KFold(n_splits=3, shuffle=True, random_state=seed)
results= cross_val_score(model, X_train, y_train, cv=kf, scoring= 'r2')
print(results)
print("Results: %.2f (%.2f) R2" % (results.mean()*100, results.std()))
