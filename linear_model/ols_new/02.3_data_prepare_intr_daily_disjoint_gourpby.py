import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings;warnings.simplefilter("ignore", category=FutureWarning)
from os import listdir;from os.path import isfile, join


import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/Volume-Forecasting/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    path = "/home/kanli/seventh/"
else:print("Unknown operating system")
data_path = path + "01_raw/"
out_path = path + "02_raw_component/"

'''New Added'''
onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
for i in tqdm(range(len(onlyfiles))):
    file = onlyfiles[i]
    df = pd.read_pickle(data_path + file)

    def aggregate_info(df,option):
        gpd = df.groupby(option).mean()[['ntn', 'volBuyNotional',
           'volSellNotional', 'nrTrades', 'ntr', 'volBuyNrTrades_lit',
           'volSellNrTrades_lit', 'volBuyQty', 'volSellQty', 'qty']]
        if option == ['date']:
            gpd = gpd.add_prefix('daily_')
        elif option == ['date','intrSn']:
            gpd = gpd.add_prefix('intraday_')
        else: raise NotImplementedError
        # gpd.insert(0,'symbols',file[:-4])
        return gpd
    tobe_appended_daily    = aggregate_info(df, ['date']) # daily info
    tobe_appended_intraday = aggregate_info(df, ['date','intrSn']) # daily info
    appended_daily = df.merge(tobe_appended_daily, on=['date'], suffixes=['', '_grouped'])
    appended_daily_intraday = appended_daily.merge(tobe_appended_intraday, on=['date','intrSn'], suffixes=['', '_grouped'])
    columns = ['ntn',
       'volBuyNotional', 'volSellNotional', 'nrTrades', 'ntr',
       'volBuyNrTrades_lit', 'volSellNrTrades_lit', 'volBuyQty', 'volSellQty']

    basic = ['symbol', 'date', 'timeHMs', 'timeHMe', 'intrSn']
    new = appended_daily_intraday[basic + columns]
    df2 = new[columns].rolling(window=2).sum()
    df8 = new[columns].rolling(window=8).sum()
    df2_columns = [col+"_2" for col in columns]
    df8_columns = [col+"_8" for col in columns]
    df2.columns = df2_columns
    df8.columns = df8_columns


    merged2 = pd.merge(appended_daily_intraday.reset_index(),df2.reset_index(), how = 'inner')
    merged8 = pd.merge(merged2,df8.reset_index(), how = 'inner')
    returned = merged8.dropna(axis=0)
    appended_daily_intraday = returned.set_index("index")
    qty = appended_daily_intraday.pop('qty')
    vo = appended_daily_intraday.pop('VO')
    appended_daily_intraday.insert(len(appended_daily_intraday.columns), 'qty', qty)
    appended_daily_intraday.insert(len(appended_daily_intraday.columns), 'VO', vo)

    try:appended_daily_intraday.to_pickle(out_path + file)
    except:
        import os;os.mkdir(out_path)
        appended_daily_intraday.to_pickle(out_path + file)



