import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from os import listdir;from os.path import isfile, join
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error as mse
plt.rcParams["figure.figsize"] = (40,20)

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

'''on mac'''
path = "/Users/kang/Data/"
data_path = path + "out_jump/"
out_path = path + 'out_ols/'
'''on stats'''
# path = "/data/cholgpu01/not-backed-up/datasets/graf/data/"
# data_path = path + "out_jump/"
'''on linux'''
# path = "/home/kanli/forth/"
# data_path = path + "out_jump/"
# out_path = path + 'out_ols/'

onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
already_done = sorted([f for f in listdir(out_path) if isfile(join(out_path, f))])

result_lst = []
# for j in tqdm(range(1,100)): # on linux1
# for j in tqdm(range(100,200)): # on linux2
# for j in tqdm(range(200,300)): # on linux3
# for j in tqdm(range(300,400)): # on mac1
# for j in tqdm(range(400,498)): # on mac2
# for j in tqdm(range(497,399,-1)): # on mac3
for j in tqdm(range(399,299,-1)): # on mac4
    file = onlyfiles[j]
    if file in already_done:
        continue
    print(f">>>> {j}th {file}")
    df = pd.read_csv(data_path + file, index_col=0)

    df_list = []
    gpd = df.groupby("date")
    for index, item in gpd:
        item['VO'] = item.qty.shift(-1)
        item = item.dropna()
        df_list.append(item)
    dflst = pd.DataFrame(pd.concat(df_list))

    X0 = dflst.iloc[:,1:-1]
    X0 = X0[['date',"intrSn","qty","volBuyQty","volSellQty","volBuyNotional","nrTrades","is_jump"]]
    y0 = dflst.iloc[:,-1]

    window_size = 3900
    rst_lst = []
    for index in tqdm(range(window_size, X0.shape[0]), leave=False):
    # for index in range(window_size, X0.shape[0]):
        # print(index)
        X = X0.iloc[index-window_size:index,:]
        y = y0.iloc[index-window_size:index]
        X = sm.add_constant(X, has_constant='add')
        results = sm.OLS(y, X).fit()
        y_hat = results.predict([1] +list(X0.iloc[index,:]))
        y_true = y0.iloc[index]
        rst_lst.append([y_true, y_hat])


    rst = pd.DataFrame(rst_lst)
    rst.iloc[:,-1] = rst.iloc[:,-1].apply(lambda x:x[0])
    rst.columns = ["yTrue","yPred"]

    result = pd.concat([dflst.iloc[window_size:,[0,1,2,3]].reset_index().drop('index',axis=1), rst],axis=1)
    result.to_csv(out_path + file)


#     result_lst.append(result)
#
#
# df = pd.concat(result_lst)
# df.to_csv(path + "ols_mw.csv")










# plt.plot(rst.iloc[:1000,0])
# plt.plot(rst.iloc[:1000,1])
# plt.show()
# mse(rst.iloc[:,0], rst.iloc[:,1])
