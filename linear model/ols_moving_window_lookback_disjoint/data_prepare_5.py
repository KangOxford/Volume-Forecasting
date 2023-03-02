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
out_path = path + "out_disjoint5/"
onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])


for i in tqdm(range(len(onlyfiles))):
    # i = 0 #$
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
    lookback_5 = dflst.rolling(lb_size,min_periods=1).sum()
    # lookback_5.iloc[lb_size-1:,4:14]/=5
    lookback_5 /= lb_size
    for i in range(lb_size-1):
        lookback_5.iloc[i,:] *= lb_size/(i+1)
    # overlapped_lookback_window }

    # # disjoint_lookback_window {
    # lb_size = 5
    # # create a new dataframe with disjoint look_back_window
    # disjoint_lb = pd.DataFrame()
    # for i in range(lb_size):
    #     # shift the dataframe i rows down and sum the last lb_size rows
    #     disjoint_lb = pd.concat([disjoint_lb, dflst.shift(i).rolling(lb_size, min_periods=1).sum()], axis=1)
    # # calculate the disjoint look_back_window by dividing the sum by lb_size
    # disjoint_lb = disjoint_lb.iloc[lb_size - 1:, ::lb_size] / lb_size
    # # adjust the first lb_size-1 rows
    # for i in range(lb_size - 1):
    #     disjoint_lb.iloc[i, :] *= lb_size / (i + 1)
    # # use the disjoint look_back_window
    # lookback_5 = disjoint_lb
    # # disjoint_lookback_window }

    append_part = lookback_5.iloc[:,4:14]
    append_part.columns = ["ol_lb5_"+ col for col in append_part.columns]

    appended = pd.concat([dflst.iloc[:,:-4], append_part, dflst.iloc[:,-4:]],axis=1)


# df = pd.DataFrame({'A': [1.1, 1.2, 1.3, 1.4, 1.5,1.6, 1.7, 1.8, 1.9, 2.0], 'B': [1.1, 1.1, 1.1, 1.1, 1.1,1.1, 1.1, 1.1, 1.1, 1.1]})
# lb_size = 3
# # rolling_sum = df.rolling(lb_size, min_periods=1).sum()
# rolling_sum = df.shift(1).rolling(lb_size).sum()










