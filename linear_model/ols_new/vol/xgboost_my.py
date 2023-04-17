from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
from itertools import product
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline
import pandas as pd
import numpy as np
from tqdm import tqdm
from os import listdir;from os.path import isfile, join
from numba import jit
from sklearn.preprocessing import StandardScaler
import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/Volume-Forecasting/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    # '''on server'''
    path = "/home/kanli/seventh/"
else:print("Unknown operating system")
data_path = path + "02_raw_component/"
onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])

import seaborn as sns

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings


warnings.filterwarnings("ignore")

if __name__=="__main__":
    df_list = []
    for item in onlyfiles:
        df = pd.read_pickle(data_path+item)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(df.iloc[:,4:-1])
        df.iloc[:,4:-1] = scaler.transform(df.iloc[:,4:-1])
        df_list.append(df)
    data = pd.concat(df_list)
    from sklearn.model_selection import train_test_split

    # Extract feature and target arrays
    X, y = data.drop('VO', axis=1), data[['VO']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    import xgboost_my as xgb
    # Create regression matrices
    dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)
    # import numpy as np
    # mse = np.mean((actual - predicted) ** 2)
    # rmse = np.sqrt(mse)
    params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}
    # Define hyperparameters
    params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}
    n = 100
    model = xgb.train(
        params=params,
        dtrain=dtrain_reg,
        num_boost_round=n,
    )
    from sklearn.metrics import mean_squared_error
    preds = model.predict(dtest_reg)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"RMSE of the base model: {rmse:.3f}")
    params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}
    n = 100
    # evals = [(dtrain_reg, "train"), (dtest_reg, "validation")
    # evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]
    #
    # model = xgb.train(
    #    params=params,
    #    dtrain=dtrain_reg,
    #    num_boost_round=n,
    #    evals=evals,
    # )
