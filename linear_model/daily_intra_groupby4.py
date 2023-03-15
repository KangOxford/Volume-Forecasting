import pandas as pd
from tqdm import tqdm
from os import listdir;from os.path import isfile, join
import warnings;warnings.simplefilter("ignore", category=FutureWarning)



import platform # Check the system platform
if platform.system() == 'Darwin':
    print("Running on MacOS")
    path = "/Users/kang/Volume-Forecasting/"
    # data_path = path + "2017/"
    # out_path = path + 'raw_older/'
elif platform.system() == 'Linux':
    print("Running on Linux")
    # # '''on server'''
    # path = "/data/cholgpu01/not-backed-up/datasets/graf/data/"
    # data_path = path + "Minutely_LOB_2017-19/"
    # out_path = path + 'out_jump/'
    path = "/home/kanli/fifth/"
    data_path1 = path + "raw_component15/"
    data_path2 = path + "out_15min_component/"
    # out_path = path + 'raw15_append4/'
else:print("Unknown operating system")
onlyfiles1 = sorted([f for f in listdir(data_path1) if isfile(join(data_path1, f))])
onlyfiles2 = sorted([f for f in listdir(data_path2) if isfile(join(data_path2, f))])
for i in range(len(onlyfiles2)):
    file1 = pd.read_pickle(data_path1+onlyfiles1[i])
    file2 = pd.read_pickle(data_path2+onlyfiles2[i])
