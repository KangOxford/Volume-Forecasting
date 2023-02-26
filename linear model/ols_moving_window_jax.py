import pandas as pd
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
from jax import jit
from jax.scipy import stats
from os import listdir
from os.path import isfile, join

@jit
def ols(train, test):
    def out_of_sample(results, test):
        pred_x = test[0]
        pred_x = jnp.hstack([jnp.ones((pred_x.shape[0], 1)), pred_x])
        pred_y = jnp.dot(pred_x, results)
        real_y = test[1]
        from sklearn.metrics import r2_score
        r_squared = r2_score(real_y, pred_y)
        return r_squared

    X = train[0]
    X = jnp.hstack([jnp.ones((X.shape[0], 1)), X])
    Y = train[1]
    results = stats.linregress(X, Y)
    print(results)
    return out_of_sample(results.slope, test)

path = "/Users/kang/Data/"
data_path = path + "out_jump/"
out_path = path + 'out_ols_jax/'

onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
already_done = sorted([f for f in listdir(out_path) if isfile(join(out_path, f))])

result_lst = []

for j in tqdm(range(399,299,-1)):
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
        X = X0.iloc[index-window_size:index,:].to_numpy()
        y = y0.iloc[index-window_size:index].to_numpy()
        results = stats.linregress(X, y)
        y_hat = jnp.dot(jnp.hstack([1, X0.iloc[index,:]]), results.slope)
        y_true = y0.iloc[index]
        rst_lst.append([y_true, y_hat])

    rst = pd.DataFrame(rst_lst)
    rst.iloc[:,-1] = rst.iloc[:,-1].apply(lambda x:x[0])
    rst.columns = ["yTrue","yPred"]

    result = pd.concat([dflst.iloc[window_size:,[0,1,2,3]].reset_index().drop('index',axis=1), rst],axis=1)
    result.to_csv(out_path + file)
