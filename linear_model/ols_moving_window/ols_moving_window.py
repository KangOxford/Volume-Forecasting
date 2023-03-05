import pandas as pd
from tqdm import tqdm
from os import listdir;from os.path import isfile, join
import statsmodels.api as sm


path = "/home/kanli/forth/"
data_path, out_path = path + "out_jump/", path + 'out_ols/'

onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
already_done = sorted([f for f in listdir(out_path) if isfile(join(out_path, f))])
for j in tqdm(range(496,400,-1)):
    file = onlyfiles[j]
    if file in already_done:
        continue
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


