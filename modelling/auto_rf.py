#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:43:58 2022

@author: kang
"""
# Tuning Random Forest Parameters using Grid Search
# -*- coding: utf-8 -*-
# import pandas as pd
import numpy as np
# from scipy.stats import shapiro
# import math
# import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# import xgboost as xgb
from sklearn.metrics import r2_score

from data_processing.data_pipeline import get_data
from data_processing.data_pipeline import overlap
from data_processing.data_pipeline import disjoint
from feature_engineering.utils import data_split


if __name__ == "__main__":
    # data,_ = get_data(1)
    # data,_ = get_data(5)
    # data = overlap("1_5")
    # data = overlap("1_5_10")
    # data = disjoint("1_5")
    # data = disjoint("1_5_10")
    
    train, test = data_split(data,size = 10)
    rf = RandomForestRegressor(n_estimators = 300, max_features=5)
    param_grid = {
    'max_features': [3, 4, 5,],
    'n_estimators': [50, 100, 200, 300, 400, 500]
    }
    
    grid_search = GridSearchCV(estimator = rf, \
        param_grid = param_grid, )
    X = train[0]
    Y = train[1]
    
    grid_search.fit(X,Y)
    print(f">>> grid_search.best_params_:{grid_search.best_params_}")
    
    rf = RandomForestRegressor(n_estimators = grid_search.best_params_['n_estimators'], \
                               max_features = grid_search.best_params_['max_features'])
    rf.fit(X,np.array(Y))
    X1 = test[0]
    real_y = test[1]
    preds = rf.predict(X1)
    r_squared = r2_score(real_y, preds)
    print('r_square_value :',r_squared)