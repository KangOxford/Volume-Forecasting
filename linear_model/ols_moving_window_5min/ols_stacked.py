import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from os import listdir;from os.path import isfile, join
from sklearn.model_selection import train_test_split

'''1. transfer data format from format A to B'''
'''2. fulfill nan with the mean of surrounding values'''
'''3. generate sym.csv with all the dates merged in single csv'''

'''format A
['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'timeIndex', 'timeHMs',
       'timeHMe', 'volBuyQty', 'volBuyNotional', 'volSellQty',
       'volSellNotional', 'nrTrades', 'volBuyQty_lit', 'volBuyNotional_lit',
       'volBuyNrTrades_lit', 'volSellQty_lit', 'volSellNotional_lit',
       'volSellNrTrades_lit', 'volBuyQty_hid', 'volBuyNotional_hid',
       'volBuyNrTrades_hid', 'volSellQty_hid', 'volSellNotional_hid',
       'volSellNrTrades_hid', 'bidPx', 'askPx', 'bidQty', 'askQty', 'pret_1m',
       'symbol', 'vol', 'jump_value', 'is_jump', 'signed_jump']
'''
'''format B
['symbol', 'date', 'timeHMs', 'timeHMe', 'intrSn', 'qty', 'volBuyQty',
       'volSellQty', 'ntn', 'volBuyNotional', 'volSellNotional', 'nrTrades',
       'ntr', 'volBuyNrTrades_lit', 'volSellNrTrades_lit']
'''

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


path = "/home/kanli/forth/"
# data_path = path + "out/"
data_path = path + "out_jump/"
onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
file = onlyfiles[0]
df = pd.read_csv(data_path + file, index_col=0)


def process_group(group):
    group['VO'] = group.qty.shift(-1)
    group = group.dropna()
    return group

# Apply the function to each group of the DataFrame and concatenate the results
df_list = df.groupby("date").apply(lambda x: process_group(x)).tolist()
dflst = pd.concat(df_list)



df_list = []
gpd = df.groupby("date")
for index, item in gpd:
    item['VO'] = item.qty.shift(-1)
    item = item.dropna()
    df_list.append(item)
dflst = pd.DataFrame(pd.concat(df_list))






X = dflst.iloc[:,1:-1]
X = X[['date',"intrSn","qty","volBuyQty","volSellQty","volBuyNotional","nrTrades","is_jump"]]
# X = X[['date',"intrSn","qty","volBuyQty","volSellQty","volBuyNotional","nrTrades"]]
y = dflst.iloc[:,-1]

oos_lst = []
for state in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=state)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    out_of_sample = ols((X_train, y_train), (X_test, y_test))
    print(f">>> out_of_sample: {out_of_sample}")
    oos_lst.append(out_of_sample)
np.mean(oos_lst)


