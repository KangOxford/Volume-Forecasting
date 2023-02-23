import pandas as pd
import numpy as np

data_path = "/Users/kang/Data/2017/"
date = "20170710"
sym = "AAPL"

df = pd.read_csv(data_path+date+'/'+date + '-'+ sym+'.csv')[['timeIndex', 'timeHMs', 'timeHMe', 'volBuyQty', 'volBuyNotional',
       'volSellQty', 'volSellNotional', 'nrTrades', 'bidPx', 'askPx', 'bidQty',
       'askQty', 'pret_1m', 'symbol', 'vol']]

from os import listdir
mypath = data_path
onlyfiles = [f for f in listdir(mypath)]
pd.DataFrame(onlyfiles).to_csv(data_path+'trading_days2017.csv')

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
