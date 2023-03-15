import pandas as pd
from tqdm import tqdm
from os import listdir;from os.path import isfile, join
import warnings;warnings.simplefilter("ignore", category=FutureWarning)



import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/Volume-Forecasting/"
    data_path = path + "2017/"
    out_path = path + 'raw_older/'
elif platform.system() == 'Linux':
    print("Running on Linux")
    # # '''on server'''
    # path = "/data/cholgpu01/not-backed-up/datasets/graf/data/"
    # data_path = path + "Minutely_LOB_2017-19/"
    # out_path = path + 'out_jump/'
    path = "/home/kanli/fifth/"
    data_path = path + "raw15/"
    out_path = path + 'raw15_append4/'
else:print("Unknown operating system")
onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])


for i in tqdm(range(len(onlyfiles))):
    file = onlyfiles[i]
    dflst = pd.read_pickle(data_path + file)

    # overlapped_lookback_window {
    lb_size = 4
    lookback_4 = dflst.shift(1).rolling(lb_size,min_periods=1).sum()
    lookback_4 /= lb_size
    for i in range(1,lb_size):
        lookback_4.iloc[i,:] *= lb_size/i
    # overlapped_lookback_window }


    append_part = lookback_4.iloc[:,4:14]
    append_part.columns = ["ol_lb5_"+ col for col in append_part.columns]

    appended = pd.concat([dflst.iloc[:,:-4], append_part, dflst.iloc[:,-4:]],axis=1)
    appended = appended.dropna(axis=0)
    try:
        appended.to_pickle(out_path + file[:-4] + '.pkl')
    except:
        import os;os.mkdir(out_path)
        appended.to_pickle(out_path + file[:-4] + '.pkl')


# df = pd.DataFrame({'A': [1.1, 1.2, 1.3, 1.4, 1.5,1.6, 1.7, 1.8, 1.9, 2.0], 'B': [1.1, 1.1, 1.1, 1.1, 1.1,1.1, 1.1, 1.1, 1.1, 1.1]})
# lb_size = 3
# lookback_5 = df.shift(1).rolling(lb_size, min_periods=1).sum()
# lookback_5 /= lb_size
# for i in range(1, lb_size):
#     lookback_5.iloc[i, :] *= lb_size / i
# # rolling_sum = df.shift(1).rolling(lb_size).sum()










