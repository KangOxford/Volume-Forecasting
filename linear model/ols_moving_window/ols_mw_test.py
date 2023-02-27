import pandas as pd
from os import listdir;from os.path import isfile, join
from sklearn.metrics import mean_squared_error


path = "/home/kanli/forth/"; data_path, out_path = path + "out_ols/", path + 'out_ols/'
onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
df_lst = pd.concat(map(lambda file: pd.read_csv(data_path + file, index_col=0), onlyfiles))
mse_date = df_lst.groupby('date').apply(lambda x: mean_squared_error(x['yTrue'],x['yPred']))
