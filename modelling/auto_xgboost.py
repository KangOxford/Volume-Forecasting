#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:43:58 2022

@author: kang
"""
# Tuning Random Forest Parameters using Grid Search
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

from data_processing.data_pipeline import get_data
from data_processing.data_pipeline import overlap
from data_processing.data_pipeline import disjoint
from feature_engineering.utils import data_split
def xgboost(
              learning_rate,
              colsample_bytree,
              silent=True,
              nthread=-1):
    train, _ = data_split(data,size = 10)
    import warnings;warnings.filterwarnings("ignore")
    score= cross_val_score(xgb.XGBRegressor(objective='reg:linear',
                                             learning_rate=learning_rate,
                                             colsample_bytree=colsample_bytree),
                           train[0],
                           train[1],
                           ).mean()
    score=np.array(score)
    return score

# if __name__ == "__main__":
    # data,_ = get_data(1)
    # data,_ = get_data(5)
    # data = overlap("1_5")
    # data = overlap("1_5_10")
    data = disjoint("1_5")
    # data = disjoint("1_5_10")
    
    xgboostBO = BayesianOptimization(xgboost,
                                 {
                                  'learning_rate': (0.01, 0.3),
                                  'colsample_bytree' :(0.5, 0.99)
                                  })
    xgboostBO.maximize(init_points=3, n_iter=50)
    
    
    
    # -------------------------
    train, test = data_split(data,size = 10)
    # learning_rate,colsample_bytree = 0.9833, 0.01996 
    # learning_rate,colsample_bytree = 0.6743, 0.02453 
    learning_rate,colsample_bytree =  0.555     , 0.02073
    X = train[0]
    Y = train[1]
    rf = xgb.XGBRegressor(objective='reg:linear',
                                              learning_rate=learning_rate,
                                              colsample_bytree=colsample_bytree)
    rf.fit(X,np.array(Y))
    X1 = test[0]
    real_y = test[1]
    preds = rf.predict(X1)
    r_squared = r2_score(real_y, preds)
    print('r_square_value :',r_squared)