import pandas as pd
import numpy as np

path = "/Users/kang/Data/"
data_path = path + "2017/"
trading_dates = pd.read_csv(path+"trading_days2017.csv",index_col=0)['0'].apply(str)
removed_dates = pd.read_csv(path+"removed_days2017.csv",index_col=0)['0'].apply(str)
dates = pd.DataFrame({'date':list(set(trading_dates.values).difference(set(removed_dates.values)))}).sort_values('date').reset_index().drop('index',axis=1)['date'].apply(str)
syms = pd.read_csv(path+"symbols.csv",index_col=0)['0'].apply(str)
i,j = 1,2
date = dates.iloc[i]
sym = syms.iloc[j]

df = pd.read_csv(data_path+date+'/'+date + '-'+ sym+'.csv')
df['qty']=df.volBuyQty+df.volSellQty;df['ntn']= df.volSellNotional+df.volBuyNotional;df['ntr']=df.volBuyNrTrades_lit+df.volSellNrTrades_lit;df['date'] = date
df = df[['symbol', 'date', 'timeHMs', 'timeHMe', 'qty', 'volBuyQty','volSellQty', 'ntn', 'volBuyNotional', 'volSellNotional',  'nrTrades','ntr', 'volBuyNrTrades_lit', 'volSellNrTrades_lit']]

def resilient_window_mean_nan(sr):
    # fullfill with the surrounding 4 non-nan values
    s_ffill = sr.ffill().ffill()
    s_bfill = sr.bfill().bfill()
    s_filled = (s_ffill + s_bfill) / 2
    return s_filled

df.apply(resilient_window_mean_nan, axis = 1)
f = resilient_window_mean_nan(df.qty)
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.plot(f.values)
plt.plot(df.qty.values)
# plt.savefig('qty_fullfill.png')
plt.show()

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
