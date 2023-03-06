import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from os import listdir;from os.path import isfile, join
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


def ols(train, test):
    def out_of_sample(results, test):
        pred_x = test[0]
        pred_x = sm.add_constant(pred_x, has_constant='add')
        pred_y = results.predict(pred_x)
        real_y = test[1]
        from sklearn.metrics import r2_score
        r_squared = r2_score(real_y, pred_y)
        return r_squared

    import statsmodels.api as sm
    X = train[0]
    # X = sm.add_constant(X)
    X = sm.add_constant(X, has_constant='add')
    Y = train[1]
    results = sm.OLS(Y, X).fit()
    print(results.summary())
    return out_of_sample(results, test)


path = "/Users/kang/Volume-Forecasting/"
data_path = path + "raw_component/"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
file = onlyfiles[0]
dflst = pd.read_pickle(data_path + file)
dflst.date =dflst.date.apply(lambda x:int(x))


window_size = 3900
col_lst = []
for index in tqdm(range(1000)):
    print(f">>> index {index}")
    X = dflst.iloc[index:window_size+index,1:-1]; y = dflst.iloc[index:window_size+index,-1]
    def ols_with_summary(X,y):
        X = sm.add_constant(X, has_constant='add')
        results = sm.OLS(y, X).fit()
        # print(results.summary())
        return results
    model = ols_with_summary(X, y)
    def feature_selecting(model,X,y):
        selected_features = model.pvalues[1:].idxmax()
        while model.pvalues[selected_features] > 0.05:
            X = X.drop(selected_features, axis=1)
            if 'const' not in X.columns: X = sm.add_constant(X, has_constant='add')
            model = sm.OLS(y, X).fit()
            selected_features = model.pvalues[1:].idxmax()
        # print(model.summary())
        return model, X, y
    model, X, y = feature_selecting(model,X,y)
    col_lst.append(X.columns.to_list())

import collections
# create a flattened list of all elements in lst
flat_lst = [item for sublist in col_lst for item in sublist]
# count the frequency of each value in flat_lst
freq = collections.Counter(flat_lst)
# print the frequency counts
for key, value in freq.most_common():
    print(key, value)

keys = list(freq.keys())




X = dflst.iloc[:,1:-1]
# X = X[['date',"intrSn","qty","volBuyQty","volSellQty","volBuyNotional","nrTrades","is_jump"]]
# X = X[['date',"intrSn","qty","volBuyQty","volSellQty","volBuyNotional","nrTrades"]]
y = dflst.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
out_of_sample = ols((X_train, y_train), (X_test, y_test))


oos_lst = []
for state in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=state)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    out_of_sample = ols((X_train, y_train), (X_test, y_test))
    print(f">>> out_of_sample: {out_of_sample}")
    oos_lst.append(out_of_sample)
np.mean(oos_lst)















trading_dates = pd.read_csv(path+"trading_days2017.csv",index_col=0)['0'].apply(str)
removed_dates = pd.read_csv(path+"removed_days2017.csv",index_col=0)['0'].apply(str)
dates = pd.DataFrame({'date':list(set(trading_dates.values).difference(set(removed_dates.values)))}).sort_values('date').reset_index().drop('index',axis=1)['date'].apply(str)
trading_syms = pd.read_csv(path+"symbols.csv",index_col=0)['0'].apply(str)
removed_syms = pd.read_csv(path+"removed_syms.csv",index_col=0)['0'].apply(str)
syms = pd.DataFrame({'syms':list(set(trading_syms.values).difference(set(removed_syms.values)))}).sort_values('syms').reset_index().drop('index',axis=1)['syms'].apply(str)




df.date = pd.to_datetime(df.date)
gpd = df.set_index('date').groupby(pd.Grouper(freq='D'))
x_list, y_list = [], []
for index, data in gpd:
    x = data.iloc[:-1,-1]
    y = data.iloc[1:,-1]
    x_list.append(x); y_list.append(y)
x = pd.concat(x_list); y = pd.concat(y_list)






def data_split(data,size = 10):
    X = data.iloc[:-size,:-1]
    Y = data.iloc[:-size,-1]
    pred_x = data.iloc[-size:,:-1]
    real_y = data.iloc[-size:,-1]
    return (X,Y),(pred_x,real_y)

def ols(train, test):
    def out_of_sample(results, test):
        pred_x = test[0]
        pred_x = sm.add_constant(pred_x, has_constant='add')
        pred_y = results.predict(pred_x)
        real_y = test[1]
        from sklearn.metrics import r2_score
        r_squared = r2_score(real_y, pred_y)
        return r_squared

    import statsmodels.api as sm
    X = train[0]
    # X = sm.add_constant(X)
    X = sm.add_constant(X, has_constant='add')
    Y = train[1]
    results = sm.OLS(Y, X).fit()
    print(results.summary())
    return out_of_sample(results, test)

    train, test = data_split(data, size=10)
    out_of_sample = ols(train, test)
    print(f">>> out_of_sample: {out_of_sample}")
