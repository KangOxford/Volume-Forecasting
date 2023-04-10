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
def optimize_SARIMA(parameters_list, d, D, s, exog):
    """
        Return dataframe with parameters, corresponding AIC and SSE

        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
        exog - the exogenous variable
    """
    results = []
    for param in tqdm_notebook(parameters_list):
        try:
            model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        results.append([param, aic])
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)x(P,Q)', 'AIC']
    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_df

if __name__=="__main__":
    df_list = []
    for item in onlyfiles:
        df = pd.read_pickle(data_path+item)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(df.iloc[:,4:-1])
        df.iloc[:,4:-1] = scaler.transform(df.iloc[:,4:-1])
        df_list.append(df)
    data = pd.concat(df_list)
    # plot_pacf(data['VO'])
    # plot_acf(data['VO'])
    # plt.show()
    ad_fuller_result = adfuller(data['VO'])
    print(f'ADF Statistic: {ad_fuller_result[0]}')
    print(f'p-value: {ad_fuller_result[1]}')
    # Indeed, the p-value is small enough for us to reject the null hypothesis,
    # and we can consider that the time series is stationary.
    p = range(0, 4, 1)
    d = 1
    q = range(0, 4, 1)
    P = range(0, 4, 1)
    D = 1
    Q = range(0, 4, 1)
    s = 4
    parameters = product(p, q, P, Q)
    parameters_list = list(parameters)
    print(len(parameters_list))
    result_df = optimize_SARIMA(parameters_list, 1, 1, 4, data['VO'])
    result_df
    best_model = SARIMAX(data['data'], order=(0, 1, 2), seasonal_order=(0, 1, 2, 4)).fit(dis=-1)
    print(best_model.summary())
    data['arima_model'] = best_model.fittedvalues
    data['arima_model'][:4 + 1] = np.NaN
    forecast = best_model.predict(start=data.shape[0], end=data.shape[0] + 8)
    forecast = data['arima_model'].append(forecast)
    plt.figure(figsize=(15, 7.5))
    plt.plot(forecast, color='r', label='model')
    plt.axvspan(data.index[-1], forecast.index[-1], alpha=0.5, color='lightgrey')
    plt.plot(data['data'], label='actual')
    plt.legend()
    plt.show()



