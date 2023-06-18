from config import *

def tryMkdir(path):
    try: listdir(path)
    except:import os;os.mkdir(path)
    return 0
_,_ = map(tryMkdir,[path01,path05])

from os import listdir;
from os.path import isfile, join;
import pandas as pd

readFromPath = lambda data_path: sorted([f for f in listdir(data_path) if isfile(join(data_path, f)) and f != '.DS_Store'])
path01Files, path04_1Files, path05Files = map(readFromPath, [path01, path04_1, path05])

for i in range(len(path04_1Files)):
    item = pd.read_csv(path04_1+path04_1Files[i],index_col=0)
    item.to_pickle(path05 + path04_1Files[i][:-3] +"pkl")
    assert item.shape[0] == 3172










