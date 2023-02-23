import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from os import listdir;from os.path import isfile, join
import statsmodels.api as s
from sklearn.metrics import mean_squared_error as mse

'''1. transfer data format from format A to B'''
'''2. fulfill nan with the mean of surrounding values'''
'''3. generate sym.csv with all the dates merged in single csv'''

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


path = "/Users/kang/Data/"
data_path = path + "out/"
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
file = onlyfiles[0]
df = pd.read_csv(data_path + file, index_col=0)

df_list = []
gpd = df.groupby("date")
for index, item in gpd:
    print
    item['VO'] = item.qty.shift(-1)
    item = item.dropna()
    df_list.append(item)
dflst = pd.DataFrame(pd.concat(df_list))

X0 = dflst.iloc[:,1:-1]
X0 = X0[['date',"intrSn","qty","volBuyQty","volSellQty","volBuyNotional","nrTrades"]]
y0 = dflst.iloc[:,-1]

window_size = 3900
rst_lst = []
for index in range(window_size, X0.shape[0]):
    print(index)
    X = X0.iloc[index-window_size:index,:]
    y = y0.iloc[index-window_size:index]
    X = sm.add_constant(X, has_constant='add')
    results = sm.OLS(y, X).fit()
    y_hat = results.predict([1] +list(X0.iloc[index,:]))
    y_true = y0.iloc[index]
    rst_lst.append([y_true, y_hat])
rst = pd.DataFrame(rst_lst)
rst.iloc[:,-1] = rst.iloc[:,-1].apply(lambda x:x[0])

# plt.plot(rst.iloc[:,0])
# plt.plot(rst.iloc[:,1])
# plt.show()
# mse(rst.iloc[:,0], rst.iloc[:,1])
