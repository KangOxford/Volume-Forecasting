import pandas as pd
from tqdm import tqdm
from os.path import isfile, join
from os import listdir, path
import statsmodels.api as sm
import numpy as np

data_path = path.join(path.expanduser("~/"), "forth", "out_jump")
out_path = path.join(path.expanduser("~/"), "forth", "out_ols_jax")
onlyfiles = sorted([f for f in listdir(data_path) if isfile(path.join(data_path, f))])

# Define a function to calculate OLS coefficients for a window of data
def calculate_ols(X, y):
    X = sm.add_constant(X, has_constant="add")
    results = sm.OLS(y, X).fit()
    return results.params

# Define a function to predict using OLS coefficients and new data
def predict(coefficients, x):
    return np.dot(coefficients, x)

for j in tqdm(range(399,299,-1)):
    file = onlyfiles[j]
    df = pd.read_csv(path.join(data_path, file), index_col=0)

    # Replace loop over groups with a single apply call
    df_list = df.groupby("date").apply(lambda x: x.assign(VO=x.qty.shift(-1)).dropna())

    # Use applymap to select columns and apply shift to qty
    X0 = df_list.iloc[:, :-1].applymap(lambda x: x.shift(1) if x.name == "qty" else x)
    X0 = X0[["date", "intrSn", "qty", "volBuyQty", "volSellQty", "volBuyNotional", "nrTrades", "is_jump"]]
    y0 = df_list.iloc[:, -1]

    window_size = 3900

    # Vectorize the calculation of OLS coefficients for all windows of data
    X = np.array([X0.iloc[i:i+window_size, :-1].values for i in range(X0.shape[0] - window_size)])
    y = np.array([y0.iloc[i:i+window_size].values for i in range(y0.shape[0] - window_size)])
    coefficients = np.vectorize(calculate_ols)(X, y)

    # Use apply to predict using OLS coefficients and new data for all windows
    preds = X0.iloc[window_size:].apply(lambda x: predict(coefficients[:, -1], np.concatenate([x.values, [1]])), axis=1)

    # Combine the predictions with the true values and save the results to a file
    rst = pd.DataFrame({"yTrue": y0[window_size:], "yPred": preds})
    result = pd.concat([df_list.iloc[window_size:, :4].reset_index(drop=True), rst], axis=1)
    result.to_csv(path.join(out_path, file))
