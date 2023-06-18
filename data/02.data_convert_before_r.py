import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings;warnings.simplefilter("ignore", category=FutureWarning)
from os import listdir;from os.path import isfile, join
from data import Config

pd.set_option('display.max_columns', None)

import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    data_path = Config.stock_merged_data_path
    out_path = Config.r_data_path
    try:
        listdir(out_path)
    except:
        import os;os.mkdir(out_path)
elif platform.system() == 'Linux':
    print("Running on Linux")
    raise NotImplementedError
else:print("Unknown operating system")

onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
for i in tqdm(range(len(onlyfiles))):
    file = onlyfiles[i]
    df = pd.read_pickle(data_path + file)
    def only_26bins(df):
        item_lst = []
        igpd = iter(df.groupby("date"))
        while True:
            try:
                item =next(igpd)[1]
                if item.shape[0] == 26:
                    item_lst.append(item)
                else:
                    pass
            except:
                break
        new_df = pd.concat(item_lst,axis=0)
        return new_df
    df1 = only_26bins(df)
    bins = np.tile(np.arange(1,27),df1.shape[0]//26)
    df1['bin'] = bins
    df1['turnover'] = df1.qty
    columns_chosen = ['date','bin','turnover','vwap_price']
    df2 = df1[columns_chosen]
    lines = ['date\tbin\tturnover\tprice\n']
    for _, row in df2.iterrows():
        line = str(row[0])+"\t"+str(row[1])+'\t'+str(row[2])+'\t'+str(row[3])+'\n'
        lines.append(line)
    symbol = file[:-4]
    write_file = symbol + '.txt'
    with open(out_path+write_file,"w+") as f:
        f.writelines(lines)




