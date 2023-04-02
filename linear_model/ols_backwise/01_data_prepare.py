import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings;warnings.simplefilter("ignore", category=FutureWarning)
from os import listdir;from os.path import isfile, join

pd.set_option('display.max_columns', None)

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
    data_path = path + "2017/"
    out_path = path + '01_raw/'
elif platform.system() == 'Linux':
    print("Running on Linux")
    # # '''on server'''
    # path = "/data/cholgpu01/not-backed-up/datasets/graf/data/"
    # data_path = path + "Minutely_LOB_2017-19/"
    # out_path = path + 'out_jump/'
    # path = "/home/kanli/fifth/"
    # data_path = path + "2017/"
    # out_path = path + 'raw15/'
    path = "/home/kanli/fifth/"
    data_path = path + "out_15min/"
    out_path = path + 'out_15min_component/'
else:print("Unknown operating system")




trading_dates = pd.read_csv(path+"trading_days2017.csv",index_col=0)['0'].apply(str)
removed_dates = pd.read_csv(path+"removed_days2017.csv",index_col=0)['0'].apply(str)
dates = pd.DataFrame({'date':list(set(trading_dates.values).difference(set(removed_dates.values)))}).sort_values('date').reset_index().drop('index',axis=1)['date'].apply(str)
trading_syms = pd.read_csv(path+"symbols.csv",index_col=0)['0'].apply(str)
removed_syms = pd.read_csv(path+"removed_syms.csv",index_col=0)['0'].apply(str)
syms = pd.DataFrame({'syms':list(set(trading_syms.values).difference(set(removed_syms.values)))}).sort_values('syms').reset_index().drop('index',axis=1)['syms'].apply(str)
try:already_done = [f[:-4] for f in listdir(out_path) if isfile(join(out_path, f))]
except:import os;os.mkdir(out_path);already_done = [f[:-4] for f in listdir(out_path) if isfile(join(out_path, f))]



# for i in tqdm(range(len(syms))):
for i in tqdm(range(10)):
    sym = syms.iloc[i]
    print(f">>> stock {i} {sym}")
    df_list = []
    for j in range(len(dates)):
        date = dates.iloc[j]
        df = pd.read_csv(data_path+date+'/'+date + '-'+ sym+'.csv')
        df['qty']=df.volBuyQty+df.volSellQty;df['ntn']= df.volSellNotional+df.volBuyNotional;df['ntr']=df.volBuyNrTrades_lit+df.volSellNrTrades_lit;df['date'] = date
        df['intrSn'] = df.timeHMs.apply(lambda x: 0 if x< 1000 else( 2 if x>=1530 else 1))
        # df = df[['symbol', 'date', 'timeHMs', 'timeHMe', 'intrSn', 'ntn', 'volBuyNotional', 'volSellNotional',  'nrTrades','ntr', 'volBuyNrTrades_lit', 'volSellNrTrades_lit', 'jump_value', 'is_jump', 'signed_jump', 'volBuyQty','volSellQty','qty']]
        df = df[['symbol', 'date', 'timeHMs', 'timeHMe', 'intrSn', 'ntn', 'volBuyNotional', 'volSellNotional',  'nrTrades','ntr', 'volBuyNrTrades_lit', 'volSellNrTrades_lit', 'volBuyQty','volSellQty','qty']]

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
        # df["VO"] = df.qty.shift(-1)
        df_list.append(df)
    dflst = pd.concat(df_list)
    dflst = dflst.reset_index().iloc[:,1:]
    dflst = dflst.reset_index()
    # dflst.iloc[:,0] = dflst.iloc[:,0]//15

    dflst['groupper'] = dflst.timeHMs.apply(lambda x:str(x).zfill(4)).apply(lambda x: x[:2]+":"+x[2:])
    dflst['groupper'] = dflst.date.apply(lambda x:str(x[:4])+'-'+str(x[4:6])+'-'+str(x[6:])+ " ") + dflst['groupper']
    dflst['groupper'] = pd.to_datetime(dflst['groupper'])

    dflst.set_index('groupper', inplace=True)
    gpd = dflst.groupby(pd.Grouper(freq='15Min'))

    dfsum = gpd.sum().dropna(axis =0)
    dfmin = gpd.min().dropna(axis =0)
    dfmax = gpd.max().dropna(axis =0)
    value = dfsum.iloc[:,4:]
    index1 = dfmin.iloc[:,[1,2,3]]
    index2 = dfmax.iloc[:,[4,5]]
    df = pd.concat([index1, index2, value],axis=1).dropna(axis =0)

    df["VO"] = df.qty.diff(1).shift(-1)/df.qty*100
    # df["VO"] = df.qty.shift(-1)
    df = df.dropna(axis=0)
    df.to_pickle(out_path+sym+'.pkl')











