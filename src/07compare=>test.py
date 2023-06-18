import numpy as np
import pandas as pd
from os import listdir;
from os.path import isfile, join;

from config import *


def tryMkdir(path):
    try: listdir(path)
    except:import os;os.mkdir(path)
    return 0
_,_ = map(tryMkdir,[path05,path06])

readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f)) and f != '.DS_Store'])
path01Files, path01_1Files, path02Files, path04Files, path05Files, path06Files =\
    map(readFromPath, [path01, path01_1, path02, path04, path05, path06])

array1 = np.concatenate( [np.arange(1,10,0.01), np.arange(10,50,0.1) ])
array2 = np.arange(1,0.001,-0.001)
combined_array = np.array(list(zip(array1, array2))).flatten()
# used for alphas

def regularity_ols(X_train, y_train, X_test, regulator):
    if regulator == "None":
        y_pred = X_test.to_numpy().flatten()
        return y_pred
    elif regulator == "OLS":
        # print("OLS")
        import statsmodels.api as sm
        def ols_with_summary(X, y):
            X = sm.add_constant(X, has_constant='add')
            results = sm.OLS(y, X).fit()
            return results

        model = ols_with_summary(X_train, y_train)
        X = sm.add_constant(X_test, has_constant='add')
        y_pred = model.predict(X).values
        # assert type(y_pred) == np.float64
        return y_pred
    elif regulator in ["Lasso", "Ridge"]:
        # print("LASSO / RIDGE")
        def find_best_regularity_alpha(X_train, y_train):
            if regulator == "Lasso":
                from sklearn.linear_model import LassoCV
                model = LassoCV(random_state=0, max_iter=10000000)
            if regulator == "Ridge":
                from sklearn.linear_model import RidgeCV
                model = RidgeCV(alphas=combined_array)
            model.fit(X_train, y_train)
            return model.alpha_

        best_regularity_alpha = find_best_regularity_alpha(X_train, y_train)
        # print(best_regularity_alpha) #$
        if regulator == "Lasso":
            from sklearn.linear_model import Lasso
            reg = Lasso(alpha=best_regularity_alpha, max_iter=10000000, tol=1e-2)
        if regulator == "Ridge":
            from sklearn.linear_model import Ridge
            reg = Ridge(alpha=best_regularity_alpha, max_iter=10000000, tol=1e-2)
        reg.fit(X_train, y_train)
        # X = pd.DataFrame(X_test).T
        # y_pred = reg.predict(X)
        y_pred = reg.predict(X_test)
        y_pred = y_pred.flatten()
        return y_pred
    else:
        raise NotImplementedError


dflst = []
for i in range(100):
    print(f">>> i: {i}")
    df = pd.read_pickle(path06+path06Files[i])
    symbol = path06Files[i][:-4]
    bin_size = 26
    train_size = 10 * 26
    test_size = 1 * 26
    index_max  = int((df.shape[0] -(train_size + test_size))/bin_size)
    r2_list = []
    # index = 0 for index in range(0, index_max+1)
    # index = 0 for index in range(0, index_max+0) # not sure

    # x_list = ['x', 'eta*seas', 'eta', 'seas', 'mu']
    # y_list = ['turnover']

    # x_list = ['eta','seas','mu']
    # y_list = ['turnover']

    our_log_features = ['log_ntn', 'log_volBuyNotional', 'log_volSellNotional', 'log_nrTrades', 'log_ntr',
                        'log_volBuyNrTrades_lit', 'log_volSellNrTrades_lit', 'log_volBuyQty', 'log_volSellQty',
                        'log_daily_ntn', 'log_daily_volBuyNotional', 'log_daily_volSellNotional', 'log_daily_nrTrades',
                        'log_daily_ntr', 'log_daily_volBuyNrTrades_lit', 'log_daily_volSellNrTrades_lit',
                        'log_daily_volBuyQty', 'log_daily_volSellQty', 'log_daily_qty', 'log_intraday_ntn',
                        'log_intraday_volBuyNotional', 'log_intraday_volSellNotional', 'log_intraday_nrTrades',
                        'log_intraday_ntr', 'log_intraday_volBuyNrTrades_lit', 'log_intraday_volSellNrTrades_lit',
                        'log_intraday_volBuyQty', 'log_intraday_volSellQty', 'log_intraday_qty', 'log_ntn_2',
                        'log_volBuyNotional_2', 'log_volSellNotional_2', 'log_nrTrades_2', 'log_ntr_2',
                        'log_volBuyNrTrades_lit_2', 'log_volSellNrTrades_lit_2', 'log_volBuyQty_2', 'log_volSellQty_2',
                        'log_ntn_8', 'log_volBuyNotional_8', 'log_volSellNotional_8', 'log_nrTrades_8', 'log_ntr_8',
                        'log_volBuyNrTrades_lit_8', 'log_volSellNrTrades_lit_8', 'log_volBuyQty_8', 'log_volSellQty_8']
    x_list = ['log_x', 'log_eta*seas', 'log_eta', 'log_seas', 'log_mu']
    x_list = x_list +  our_log_features
    y_list = ['log_turnover']
    # x_list = ['log_eta', 'log_seas', 'log_mu']
    # y_list = ['log_turnover']
    # x_list = ['x']
    # y_list = ['turnover']

    # x_list = ['log_x']
    # y_list = ['log_turnover']


    original_space = ['turnover']
    for index in range(0, index_max + 1):
        train_end_index = index * bin_size + train_size
        def get_trainData(df):
            x_train = df.loc[index * bin_size : index * bin_size + train_size,x_list]
            y_train = df.loc[index * bin_size : index * bin_size + train_size,y_list]
            return x_train, y_train
        def get_testData(df):
            x_test = df.loc[train_end_index:train_end_index+test_size,x_list]
            y_test = df.loc[train_end_index:train_end_index+test_size,y_list]
            return x_test, y_test
        X_train, y_train = get_trainData(df)
        X_test, y_test = get_testData(df)

        # regulator = "OLS"
        regulator = "Lasso"
        # regulator = "Ridge"
        # regulator = "None"
        y_pred = regularity_ols(X_train, y_train, X_test, regulator)
        min_limit, max_limit = y_train.min(), y_train.max()
        broadcast = lambda x:np.full(y_pred.shape[0], x.to_numpy())
        min_limit, max_limit= map(broadcast, [min_limit, max_limit])
        y_pred_clipped = np.clip(y_pred, min_limit, max_limit)
        if any('log' in x for x in x_list):
            y_pred_clipped = np.exp(y_pred_clipped)
        test_date = df.date[train_end_index]


        from sklearn.metrics import r2_score
        original_images = df.loc[train_end_index:train_end_index+test_size,original_space]
        # r2 = r2_score(y_test, y_pred_clipped)
        r2 = r2_score(original_images, y_pred_clipped)


        r2_list.append([test_date,r2])
        # y_list.append([test_date, y_test, y_pred_clipped])
    r2arr = np.array(r2_list)
    df = pd.DataFrame(r2arr)
    # r2arr[:,1].mean()
    df.columns = ['test_date','r2']
    df['symbol'] = symbol
    df = df[['symbol','test_date','r2']]
    df.test_date = df.test_date.astype(int)
    pivot_df = df.pivot(index='test_date', columns='symbol', values='r2')
    dflst.append(pivot_df)
r2df = pd.concat(dflst,axis =1)
r2df.to_csv(path00 + "07_r2df_"+regulator+"_.csv", mode = 'w')
