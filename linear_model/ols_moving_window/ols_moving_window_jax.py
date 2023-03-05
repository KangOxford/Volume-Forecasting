import pandas as pd
import jax.numpy as jnp
from jax import jit, vmap
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import statsmodels.api as sm

'''on linux'''
path = "/home/kanli/forth/"
data_path = path + "out_jump/"
out_path = path + 'out_ols_numba/'

onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])

@jit
def calculate_ols(X, y):
    X = sm.add_constant(X, has_constant="add")
    results = sm.OLS(y, X).fit()
    return results.params

for j in tqdm(range(1, 100)):
    file = onlyfiles[j]
    df = pd.read_csv(data_path + file, index_col=0)

    df_list = []
    gpd = df.groupby("date")
    for index, item in gpd:
        item["VO"] = item.qty.shift(-1)
        item = item.dropna()
        df_list.append(item)
    dflst = pd.DataFrame(pd.concat(df_list))

    X0 = dflst.iloc[:, 1:-1]
    X0 = X0[["date", "intrSn", "qty", "volBuyQty", "volSellQty", "volBuyNotional", "nrTrades", "is_jump"]]
    y0 = dflst.iloc[:, -1]

    window_size = 3900

    # Define a function that takes a window of data and returns a prediction
    @jit
    def predict(x):
        X = x.iloc[:, :-1]
        y = x.iloc[:, -1]
        return jnp.dot(calculate_ols(X, y), x.iloc[-1, :-1])

    # Use vmap to apply the prediction function to each window of data
    print(">>>> before vmap")
    preds = vmap(predict)(jnp.array([X0.iloc[i : i + window_size + 1, :] for i in range(X0.shape[0] - window_size)]))
    print(">>>> after vmap")

    # Combine the predictions with the true values and save the results to a file
    rst = pd.DataFrame({"yTrue": y0[window_size:], "yPred": preds})
    result = pd.concat([dflst.iloc[window_size:, [0, 1, 2, 3]].reset_index().drop("index", axis=1), rst], axis=1)
    result.to_csv(out_path + file)
