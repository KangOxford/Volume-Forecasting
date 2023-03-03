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
    path = "/Users/kang/Desktop/Volume-Forecasting/"
    data_path = path + "out_jump/"
    out_path = path + "out_disjoint5/"
elif platform.system() == 'Linux':
    print("Running on Linux")
    # '''on server'''
    path = "/home/kanli/forth/"
    data_path = path + "out_jump/"
    out_path = path + "out_disjoint5/"
else:print("Unknown operating system")

'''New Added'''
onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
for i in tqdm(range(len(onlyfiles))):
    file = onlyfiles[i]
    df = pd.read_csv(data_path + file, index_col=0)
    df_list = []
    gpd = df.groupby("date")
    for index, item in gpd:
        item['VO'] = item.qty.shift(-1)
        item = item.dropna()
        df_list.append(item)
    dflst = pd.DataFrame(pd.concat(df_list))

    # overlapped_lookback_window {
    lb_size = 5
    lookback_5 = dflst.shift(1).rolling(lb_size,min_periods=1).sum()
    # lookback_5.iloc[lb_size-1:,4:14]/=5
    lookback_5 /= lb_size
    for i in range(1,lb_size):
        lookback_5.iloc[i,:] *= lb_size/i
    # overlapped_lookback_window }


    append_part = lookback_5.iloc[:,4:14]
    append_part.columns = ["ol_lb5_"+ col for col in append_part.columns]

    appended = pd.concat([dflst.iloc[:,:-4], append_part, dflst.iloc[:,-4:]],axis=1)
    appended = appended.dropna(axis=0)
    try:
        appended.to_pickle(out_path + file[:-4] + '.pkl')
    except:
        import os;os.mkdir(out_path)
        appended.to_pickle(out_path + file[:-4] + '.pkl')
'''New Added'''

trading_dates = pd.read_csv(path+"trading_days2017.csv",index_col=0)['0'].apply(str)
removed_dates = pd.read_csv(path+"removed_days2017.csv",index_col=0)['0'].apply(str)
dates = pd.DataFrame({'date':list(set(trading_dates.values).difference(set(removed_dates.values)))}).sort_values('date').reset_index().drop('index',axis=1)['date'].apply(str)
trading_syms = pd.read_csv(path+"symbols.csv",index_col=0)['0'].apply(str)
removed_syms = pd.read_csv(path+"removed_syms.csv",index_col=0)['0'].apply(str)
syms = pd.DataFrame({'syms':list(set(trading_syms.values).difference(set(removed_syms.values)))}).sort_values('syms').reset_index().drop('index',axis=1)['syms'].apply(str)
try:already_done = [f[:-4] for f in listdir(out_path) if isfile(join(out_path, f))]
except:import os;os.mkdir(out_path);already_done = [f[:-4] for f in listdir(out_path) if isfile(join(out_path, f))]



for i in tqdm(range(len(syms))):
    sym = syms.iloc[i]
    print(f">>> stock {i} {sym}")
    df_list = []
    for j in range(len(dates)):
        date = dates.iloc[j]

        df = pd.read_csv(data_path+date+'/'+date + '-'+ sym+'.csv')
        df['qty']=df.volBuyQty+df.volSellQty;df['ntn']= df.volSellNotional+df.volBuyNotional;df['ntr']=df.volBuyNrTrades_lit+df.volSellNrTrades_lit;df['date'] = date
        df['intrSn'] = df.timeHMs.apply(lambda x: 0 if x< 1000 else( 2 if x>=1530 else 1))
        # df = df[['symbol', 'date', 'timeHMs', 'timeHMe', 'intrSn', 'qty', 'volBuyQty','volSellQty', 'ntn', 'volBuyNotional', 'volSellNotional',  'nrTrades','ntr', 'volBuyNrTrades_lit', 'volSellNrTrades_lit']]
        df = df[['symbol', 'date', 'timeHMs', 'timeHMe', 'intrSn', 'qty', 'volBuyQty','volSellQty', 'ntn', 'volBuyNotional', 'volSellNotional',  'nrTrades','ntr', 'volBuyNrTrades_lit', 'volSellNrTrades_lit', 'jump_value', 'is_jump', 'signed_jump']]

        def resilient_window_mean_nan(sr):
            def double_fullfill(sr):
                # fullfill with the surrounding 4 non-nan values
                s_ffill = sr.ffill().ffill()
                s_bfill = sr.bfill().bfill()
                s_filled = (s_ffill + s_bfill) / 2
                return s_filled
            ffill = lambda sr: sr.ffill()
            bfill = lambda sr: sr.bfill()
            rst = double_fullfill(sr)
            rst = ffill(rst)
            rst = bfill(rst)
            return rst

        df.iloc[:,5:] = df.iloc[:,5:].apply(resilient_window_mean_nan, axis = 0)
        df_list.append(df)
    dflst = pd.concat(df_list)
    dflst.to_csv(out_path+sym+'.csv')












