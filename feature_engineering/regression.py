# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
# import pandas as pd
import numpy as np
# from scipy.stats import shapiro
# import math
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV
# import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV



from data_processing.data_pipeline import get_data
from data_processing.data_pipeline import overlap
from data_processing.data_pipeline import disjoint
from feature_engineering.utils import data_split



if __name__ == "__main__":
    # data,_ = get_data(5)
    # data = overlap("1_5")
    # data = overlap("1_5_10")
    # data = disjoint("1_5")
    data = disjoint("1_5_10")
    train, test = data_split(data,size = 10)
    clf = RidgeCV(alphas = np.arange(0.001,2,0.01),store_cv_values=True, cv=None)



    X = train[0]
    Y = train[1]
    model = clf.fit(X, Y)
    
    X1 = test[0]
    real_y = test[1]
    preds = model.predict(X1)
    r_squared = r2_score(real_y, preds)
    print('r_square_value :',r_squared)