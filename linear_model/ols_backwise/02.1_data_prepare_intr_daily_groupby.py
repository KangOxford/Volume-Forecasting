import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings;warnings.simplefilter("ignore", category=FutureWarning)
from os import listdir;from os.path import isfile, join

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

import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/Volume-Forecasting/"
    data_path = path + "01_raw/"
    out_path = path + "02_raw_component/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    # '''on server'''
    path = "/home/kanli/fifth/"
    data_path = path + "raw15/"
    out_path = path + "raw_component15/"
else:print("Unknown operating system")

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
    qty = appended_daily_intraday.pop('qty')
    vo = appended_daily_intraday.pop('VO')
    appended_daily_intraday.insert(len(appended_daily_intraday.columns), 'qty', qty)
    appended_daily_intraday.insert(len(appended_daily_intraday.columns), 'VO', vo)
    try:appended_daily_intraday.to_pickle(out_path + file)
    except:
        import os;os.mkdir(out_path)
        appended_daily_intraday.to_pickle(out_path + file)



