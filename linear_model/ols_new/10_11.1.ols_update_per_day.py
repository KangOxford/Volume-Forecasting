import pandas as pd
import numpy as np
from tqdm import tqdm
from os import listdir;from os.path import isfile, join
from numba import jit
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import Normalizer

# import warnings
# from sklearn.exceptions import ConvergenceWarning
# warnings.filterwarnings('ignore', category=ConvergenceWarning)


import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/Volume-Forecasting/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    # '''on server'''
    path = "/home/kanli/seventh/"
else:print("Unknown operating system")
data_path = path + "02_raw_component/"
out_path = path + '03_out_15min_pred_true_pairs_after_ols/'

try: listdir(out_path)
except:import os;os.mkdir(out_path)


onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
already_done = sorted([f for f in listdir(out_path) if isfile(join(out_path, f))])

@jit(nopython=True)
def ols(X, y):
    X = np.column_stack((np.ones(X.shape[0]), X))
    XT_X_pinv = np.linalg.pinv(X.T @ X)
    beta = XT_X_pinv @ X.T @ y
    return beta

array1 = np.concatenate( [np.arange(1,10,0.01), np.arange(10,50,0.1) ])
array2 = np.arange(1,0.001,-0.001)
combined_array = np.array(list(zip(array1, array2))).flatten()
# used for alphas


# regulator = "Ridge"
# regulator = "Lasso"
regulator = "OLS"

if __name__=="__main__":
    r2_score_arr_list = []
    mse_score_arr_list = []
    y_true_pred_arr_list = []
    for i in tqdm(range(len(onlyfiles))): # on mac4
        bin_size = 26
        num_day = 10
        file = onlyfiles[i]
        # if file in already_done:
        #     print(f"++++ {j}th {file} is already done before")
        #     continue
        print(f">>>> {i}th {file}")
        dflst = pd.read_pickle(data_path + file)
        counted = dflst.groupby("date").count()
        date = pd.DataFrame(counted[counted['VO'] ==26].index)
        dflst = pd.merge(dflst, date, on = "date")
        assert dflst.shape[0]/ bin_size ==  dflst.shape[0]// bin_size

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(dflst.iloc[:,4:-1])
        dflst.iloc[:,4:-1] = scaler.transform(dflst.iloc[:,4:-1])

        r2_score_list = []
        mse_score_list = []
        y_true_pred_list = []
        times = np.arange(0,dflst.shape[0]//bin_size-num_day)
        for time in times:
            index = bin_size * time
            X_train = dflst.iloc[index:index+bin_size*num_day,4:-1]
            y_train = dflst.iloc[index:index+bin_size*num_day:,-1]
            X_test = dflst.iloc[index+bin_size*num_day:index+bin_size*(num_day+1),4:-1]
            y_test = dflst.iloc[index+bin_size*num_day:index+bin_size*(num_day+1),-1].values
            def regularity_ols(X_train, y_train, X_test ,regulator):
                if regulator == "OLS":
                    # print("OLS")
                    import statsmodels.api as sm
                    def ols_with_summary(X, y):
                        X = sm.add_constant(X, has_constant='add')
                        results = sm.OLS(y, X).fit()
                        # print(results.summary())
                        return results
                    def feature_selecting(model, X, y):
                        selected_features = model.pvalues[1:].idxmax()
                        while model.pvalues[selected_features] > 0.05:
                            X = X.drop(selected_features, axis=1)
                            if 'const' not in X.columns: X = sm.add_constant(X, has_constant='add')
                            model = sm.OLS(y, X).fit()
                            selected_features = model.pvalues[1:].idxmax()
                        # print(model.summary())
                        return model, X, y
                    model = ols_with_summary(X_train, y_train)
                    model, X, y = feature_selecting(model, X_train, y_train)
                    columns = X.columns.to_list()
                    # test sets
                    if 'const' in columns:
                        columns.remove('const')
                        X = X_test[columns]
                        X = sm.add_constant(X, has_constant='add')
                    y_pred = model.predict(X)
                    return y_pred
                elif regulator in ["Lasso","Ridge"]:
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
                    print(best_regularity_alpha) #$
                    if regulator == "Lasso":
                        from sklearn.linear_model import Lasso
                        reg = Lasso(alpha=best_regularity_alpha,max_iter=10000000, tol = 1e-2)
                    if regulator == "Ridge":
                        from sklearn.linear_model import Ridge
                        reg = Ridge(alpha=best_regularity_alpha,max_iter=10000000, tol = 1e-2)
                    reg.fit(X_train, y_train)
                    y_pred = reg.predict(X_test)
                    return y_pred
                else: raise NotImplementedError
            # reg = regularity_ols(X_train, y_train, regulator)
            # y_pred = reg.predict(X_test)
            y_pred = regularity_ols(X_train, y_train, X_test, regulator)
            min_limit, max_limit = y_train.min(), y_train.max()
            y_pred = np.vectorize(lambda x: min(max(min_limit, x),max_limit))(y_pred)
            from sklearn.metrics import r2_score
            r2_score_value = r2_score(y_test,y_pred)
            from sklearn.metrics import mean_squared_error
            mse_score_value = mean_squared_error(y_test,y_pred)
            date = dflst.date.iloc[index+bin_size*num_day]
            y_true_pred = np.array([np.full(bin_size, file[:-4]).astype(str),np.full(bin_size, date).astype(str), y_test, y_pred.astype(np.float32)]).T
            y_true_pred_list.append(y_true_pred)
            r2_score_list.append([file[:-4],date,r2_score_value])
            mse_score_list.append([file[:-4],date,mse_score_value])

            # print('R squared training set', round(reg.score(X_train, y_train) * 100, 2))
            # from sklearn.metrics import mean_squared_error
            # # Training data
            # pred_train = reg.predict(X_train)
            # mse_train = mean_squared_error(y_train, pred_train)
            # print('MSE training set', round(mse_train, 2))
        # warnings.filterwarnings("default", category=RuntimeWarning)
        y_true_pred_arr = np.array(y_true_pred_list).reshape(-1,4)
        r2_score_arr = np.array(r2_score_list)
        mse_score_arr = np.array(mse_score_list)

        r2_score_arr_list.append(r2_score_arr)
        mse_score_arr_list.append(mse_score_arr)
        y_true_pred_arr_list.append(y_true_pred_arr)

    r2_score_arr_arr = np.array(r2_score_arr_list).reshape(-1,3)
    mse_score_arr_arr = np.array(mse_score_arr_list).reshape(-1,3)
    y_true_pred_arr_arr = np.array(y_true_pred_arr_list).reshape(-1,4)

    r2_score_arr_df = pd.DataFrame(r2_score_arr_arr,columns=["symbol",'date','value'])
    mse_score_arr_df = pd.DataFrame(mse_score_arr_arr,columns=["symbol",'date','value'])
    y_true_pred_arr_df = pd.DataFrame(y_true_pred_arr_arr,columns=["symbol",'date','true','pred'])

    result_data_path = path + "05_result_data_path/"+regulator+"/"
    try:listdir(result_data_path)
    except:import os;os.mkdir(result_data_path)

    r2_score_arr_df.to_csv(result_data_path + "r2_score.csv")
    mse_score_arr_df.to_csv(result_data_path + "mse_score.csv")
    y_true_pred_arr_df.to_csv(result_data_path + "y_true_pred.csv")



    array1 = np.concatenate( [np.arange(1,10,0.01), np.arange(10,50,0.1) ])
    array2 = np.arange(1,0.001,-0.001)
    combined_array = np.array(list(zip(array1, array2))).flatten()
    print(combined_array)
