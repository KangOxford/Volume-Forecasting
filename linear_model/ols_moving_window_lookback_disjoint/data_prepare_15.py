import pandas as pd
from tqdm import tqdm
from os import listdir;from os.path import isfile, join
import warnings;warnings.simplefilter("ignore", category=FutureWarning)

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

path = "/home/kanli/forth/"
data_path = path + "out_jump/"
out_path = path + "out_disjoint15/"
onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])


for i in tqdm(range(len(onlyfiles))):
    # i = 1 #$
    file = onlyfiles[i]
    df = pd.read_csv(data_path + file, index_col=0)
    df_list = []
    gpd = df.groupby("date")
    for index, item in gpd:
        item['VO'] = item.qty.shift(-1)
        item = item.dropna()
        df_list.append(item)
    dflst = pd.DataFrame(pd.concat(df_list))

    lb_size_list = [1,5,15]
    append_part_list = []
    for i in range(len(lb_size_list)-1):
        # i = 1 #$
        # i = 0 #$
        lb_size = lb_size_list[i+1]
        lookback_5 = dflst.shift(sum(lb_size_list[:i+1])).rolling(lb_size,min_periods=1).sum()

        lookback_5 /= lb_size
        for j in range(sum(lb_size_list[:i+1]),sum(lb_size_list[:i+1])+lb_size-1):
            lookback_5.iloc[j,:] *= lb_size/(j-sum(lb_size_list[:i+1])+1)
        append_part = lookback_5.iloc[:,4:14]
        append_part.columns = ["ol_lb"+str(lb_size)+"_"+ col for col in append_part.columns]
        append_part_list.append(append_part)

    appended = pd.concat([dflst.iloc[:,:-4], append_part_list[0], append_part_list[1], dflst.iloc[:,-4:]],axis=1)
    appended = appended.dropna(axis=0)
    try:
        appended.to_pickle(out_path + file[:-4] + '.pkl')
    except:
        import os;os.mkdir(out_path)
        appended.to_pickle(out_path + file[:-4] + '.pkl')
