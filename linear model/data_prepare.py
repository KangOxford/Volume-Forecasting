import pandas as pd
import numpy as np
from tqdm import tqdm

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

path = "/Users/kang/Data/"
data_path = path + "2017/"
out_path = path + 'out/'
trading_dates = pd.read_csv(path+"trading_days2017.csv",index_col=0)['0'].apply(str)
removed_dates = pd.read_csv(path+"removed_days2017.csv",index_col=0)['0'].apply(str)
dates = pd.DataFrame({'date':list(set(trading_dates.values).difference(set(removed_dates.values)))}).sort_values('date').reset_index().drop('index',axis=1)['date'].apply(str)
trading_syms = pd.read_csv(path+"symbols.csv",index_col=0)['0'].apply(str)
removed_syms = pd.read_csv(path+"removed_syms.csv",index_col=0)['0'].apply(str)
syms = pd.DataFrame({'syms':list(set(trading_syms.values).difference(set(removed_syms.values)))}).sort_values('syms').reset_index().drop('index',axis=1)['syms'].apply(str)

for i in tqdm(range(len(syms))):
    sym = syms.iloc[i]
    print(f">>> stock {i} {sym}")
    df_list = []
    for j in range(len(dates)):
        date = dates.iloc[j]

        df = pd.read_csv(data_path+date+'/'+date + '-'+ sym+'.csv')
        df['qty']=df.volBuyQty+df.volSellQty;df['ntn']= df.volSellNotional+df.volBuyNotional;df['ntr']=df.volBuyNrTrades_lit+df.volSellNrTrades_lit;df['date'] = date
        df['intrSn'] = df.timeHMs.apply(lambda x: 0 if x< 1000 else( 2 if x>=1530 else 1))
        df = df[['symbol', 'date', 'timeHMs', 'timeHMe', 'intrSn', 'qty', 'volBuyQty','volSellQty', 'ntn', 'volBuyNotional', 'volSellNotional',  'nrTrades','ntr', 'volBuyNrTrades_lit', 'volSellNrTrades_lit']]

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












