# -*- coding: utf-8 -*-

from sklearn.linear_model import RidgeCV

results = RidgeCV(alphas = np.arange(0.001,2,0.01),store_cv_values=True, cv=None).fit(, Y)