import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mypath = '/Users/kang/Minutely_LOB_2017-19'

# from os import listdir
# from os.path import isfile, join
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# lst = listdir(mypath)
# for date in lst:
#     if date != '.DS_Store':
#         os.system('cp '+ mypath+'/'+date+'/'+date+'-AMZN.csv' + " /Users/kang/Desktop/amzn_data/")


datapath = "/Users/kang/Desktop/apple_data/"
# datapath = "/data/cholgpu01/not-backed-up/datasets/graf/data/Minutely_LOB_2017-19"
datapath = "/data/cholgpu01/not-backed-up/datasets/graf/data/Minutely_LOB_2017-19/20170301"
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(datapath) if isfile(join(datapath, f))]
onlyfiles.remove('.DS_Store')
onlyfiles = sorted(onlyfiles)



# %%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
df = pd.read_csv("symbols.csv",index_col=0)
sym = df.iloc[:,0].apply(lambda x: x[9:-4])

for j in range(10):
    stock = sym[j]
    print(f">>>>stock {j}  {stock}")
    datapath = "/data/cholgpu01/not-backed-up/datasets/graf/data/Minutely_LOB_2017-19/"
    from os import listdir;import pandas as pd;import numpy as np
    dirs = [f for f in listdir(datapath)]
    dirs = sorted(dirs)
    df = pd.DataFrame(dirs)
    date = df.iloc[:, 0].values[0]
    file = pd.read_csv(datapath + date + '/' + date +'-'+stock+'.csv')[['timeHMs', 'nrTrades']]
    time_stamp = file.timeHMs.values
    df_lst = []

    for i in range(len(df)):
        date = df.iloc[:, 0].values[i]
        try:
            file = pd.read_csv(datapath + date + '/' + date + '-'+stock+'.csv')[['timeHMs', 'nrTrades']]
            file['date'] = date
            df_lst.append(file)
        except:
            pass

    dflst = pd.concat(df_lst)
    dflst = dflst.reset_index()
    dflst = dflst.drop('index',axis = 1)
    dflst.to_csv(datapath +'output/'+stock+'.csv')





























# %%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
datapath = "~/NVDA_raw1.csv"
# datapath = "~/amazon_raw1.csv"
df = pd.read_csv(datapath)
df.columns = ['index','timeHMs','nrTrades','date']
df = df.drop('index',axis = 1)

def converting(x):
    try: return float(x)
    except: return np.NaN
df.nrTrades = df.nrTrades.apply(converting)


groupped = df.groupby('timeHMs').mean()['nrTrades']
# plt.scatter(np.arange(len(groupped.index)), y = groupped.values)
# plt.show()
file = pd.DataFrame(groupped).reset_index()
file.timeHMs = file.timeHMs.apply(lambda x: str(x).rjust(4, '0'))
file.timeHMs = file.timeHMs.apply(lambda x: x[:2] + ':' + x[2:])
# file['date'] = ['20170703' ]* file.shape[0]
# file.date = file.date.apply(lambda x: str(x))
# file.date = file.date.apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:] + ' ')
# file['time'] = file.date + file.timeHMs
file['time'] =  file.timeHMs
file.time = file.time.apply(pd.to_datetime)
groupped0 = file[['time', 'nrTrades']]
groupped0['time'] = groupped0['time'].apply(lambda x: x.replace(year=2017, month = 7, day=3))
groupped0_1 = groupped0.set_index('time')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%

file = df
file.timeHMs = file.timeHMs.apply(lambda x: str(x).rjust(4, '0'))
file.timeHMs = file.timeHMs.apply(lambda x: x[:2] + ':' + x[2:])
file.date = file.date.apply(lambda x: str(x))
file.date = file.date.apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:] + ' ')
file['time'] = file.date + file.timeHMs
file.time = file.time.apply(pd.to_datetime)
df0 = file[['time', 'nrTrades']]
df0.columns = ['date', 'VO']

# import datetime as dt
# df = df[(df['date'] > '2000-6-1') & (df['date'] <= '2000-6-10')]
# df0[df0.date == dt.datetime(2017,7,3)]
from datetime import datetime
df1 = df0.set_index('date')
# df1[datetime(2017, 7, 3)]
bf0 = df1['6/30/2017':'7/1/2017']
# err0= df1['7/3/2017':'7/4/2017']
af0 = df1['7/4/2017':'7/5/2017']

avr0=(bf0.values + af0.values)/2
# avr0=(bf0.iloc[:,0].to_list()+af0.iloc[:,0].to_list())/2
def plot_(avr0):
    plt.scatter(np.arange(len(avr0.index)), avr0.values)
    plt.show()
plot_(bf0)
plot_(af0)
plot_(avr0)
df1['7/3/2017':'7/4/2017'] = avr0
plot_(df1['7/3/2017':'7/4/2017'])


bf0 = df1['11/22/2017':'11/23/2017']
af0 = df1['11/26/2017':'11/27/2017']
df1['11/23/2017':'11/24/2017'] = (bf0.values + af0.values)/2
plot_(df1['11/23/2017':'11/24/2017'])

df1.to_csv('NVDA.csv')
# df1.to_csv('NVDA.csv',index = None)



plt.scatter(np.arange(len(err0.index)), err0.values)
plt.show()
err0_1 = err0.dropna()
plt.scatter(np.arange(len(err0_1.index)), err0_1.values)
plt.show()
factor = err0_1.mean().values/groupped0_1.loc[err0_1.index].mean()
revised_value = factor * groupped0_1
# for i in range(groupped0_1.shape[0]):
nan_index = sorted(list(set(groupped0_1.index).difference( set(groupped0_1.loc[err0_1.index].index))))
assert len(nan_index) + groupped0_1.loc[err0_1.index].shape[0] == groupped0_1.shape[0]

err0.loc[nan_index] = revised_value.loc[nan_index]
plt.scatter(np.arange(len(err0.index)), y = err0.values)
plt.show()






df0.to_csv("NVDA.csv",index=None)








# %%%%%%%%%%%%%%%%%%%%%%%%%%%


datapath = "/data/cholgpu01/not-backed-up/datasets/graf/data/Minutely_LOB_2017-21/"
stock = 'NVDA'
# stock = 'AAPL'
# stock = 'AMZN'
from os import listdir;import pandas as pd;import numpy as np
dirs = [f for f in listdir(datapath)]
dirs = sorted(dirs)
df_dirs = pd.DataFrame(dirs)
# df = df_dirs[df_dirs.iloc[:,0] >='20210000']
df = df_dirs[df_dirs.iloc[:,0] <='20180000']
# df = df_dirs[df_dirs.iloc[:,0]>='20190000']
# df = df_dirs[df_dirs.iloc[:,0]<='20190000']
# df = df[df.iloc[:,0]>='20180000']
date = df.iloc[:, 0].values[0];file = pd.read_csv(datapath + date + '/' + date +'-'+stock+'.csv')[['timeHMs', 'nrTrades']]
time_stamp = file.timeHMs.values
df_lst = []

for i in range(len(df)):
    date = df.iloc[:, 0].values[i]
    try:
        file = pd.read_csv(datapath + date + '/' + date + '-'+stock+'.csv')[['timeHMs', 'nrTrades']]
        # if date in ['20170703', '20171124']:
        #     file = pd.read_csv(datapath + date + '/' + date + '-'+stock+'.csv')[['timeHMs', 'nrTrades']]
        #     wrong_stamp = file.timeHMs.values
        #     diff_stamp = list(set(time_stamp).difference(set(wrong_stamp)))
        #     adjusted = pd.DataFrame({'timeHMs': diff_stamp, 'nrTrades': np.NaN})
        #     new = pd.concat([adjusted, file])
        #     file = new.sort_values("timeHMs").reset_index().drop(['index'], axis=1)
        file['date'] = date
        df_lst.append(file)
        print(f"+ date:{date} finished")
    except:
        print(f">>>> no stored data on date:{date}")

dflst = pd.concat(df_lst)
dflst = dflst.reset_index()
dflst = dflst.drop('index',axis = 1)
# dflst = dflst.drop('index',axis = 1)
# dflst.to_csv('apple.csv')
dflst.to_csv(stock+'_raw1.csv')



# %%%%%%%%%%%%%%%%%%%%%%%%%%%

datapath = "/data/cholgpu01/not-backed-up/datasets/graf/data/Minutely_LOB_2017-21/"
from os import listdir;import pandas as pd;import numpy as np
dirs = [f for f in listdir(datapath)]
dirs = sorted(dirs)
df_dirs = pd.DataFrame(dirs)
# df = df_dirs[df_dirs.iloc[:,0] >='20210000']
df = df_dirs[df_dirs.iloc[:,0] <='20180000']
# df = df_dirs[df_dirs.iloc[:,0]>='20190000']
# df = df_dirs[df_dirs.iloc[:,0]<='20190000']
# df = df[df.iloc[:,0]>='20180000']
date = df.iloc[:, 0].values[0];file = pd.read_csv(datapath + date + '/' + date + '-AMZN.csv')[['timeHMs', 'nrTrades']]
time_stamp = file.timeHMs.values
df_lst = []

for i in range(len(df)):
    date = df.iloc[:, 0].values[i]
    try:
        file = pd.read_csv(datapath + date + '/' + date + '-AMZN.csv')[['timeHMs', 'nrTrades']]
        # file = pd.read_csv(datapath + date + '/' + date + '-AAPL.csv')[['timeHMs', 'nrTrades']]
        if date in ['20170703', '20171124']:
            file = pd.read_csv(datapath + date + '/' + date + '-AMZN.csv')[['timeHMs', 'nrTrades']]
            wrong_stamp = file.timeHMs.values
            diff_stamp = list(set(time_stamp).difference(set(wrong_stamp)))
            adjusted = pd.DataFrame({'timeHMs': diff_stamp, 'nrTrades': np.NaN})
            new = pd.concat([adjusted, file])
            file = new.sort_values("timeHMs").reset_index().drop(['index'], axis=1)
        file['group'] = file.timeHMs // 5
        groupped = file.groupby('group').sum()
        groupped.timeHMs = (groupped.timeHMs - 10) // 5
        file = groupped
        file.timeHMs = file.timeHMs.apply(lambda x: str(x).rjust(4, '0'))
        file.timeHMs = file.timeHMs.apply(lambda x: x[:2] + ':' + x[2:])
        file['date'] = date
        file.date = file.date.apply(lambda x: str(x))
        file.date = file.date.apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:] + ' ')
        file['time'] = file.date + file.timeHMs
        file.time = file.time.apply(pd.to_datetime)
        df0 = file[['time', 'nrTrades']]
        df_lst.append(df0)
        print(f"+ date:{date} finished")
    except:
        print(f">>>> no stored data on date:{date}")

dflst = pd.concat(df_lst)
dflst = dflst.reset_index();dflst = dflst.drop('group',axis = 1)
# dflst = dflst.drop('index',axis = 1)
# dflst.to_csv('apple.csv')
dflst.to_csv('amozon.csv')

# date = '20170703'
# print(file.timeHMs.values)




# %%%%%%%%%%%%%%%%%%%%%%%%%%%

df_lst = []
for file_name in onlyfiles:
    # file_name = onlyfiles[100] #$
    print(f'>>>>> {file_name}')
    if file_name in ("20180703-AAPL.csv","20170811-AAPL.csv","20181123-AAPL.csv"):
        continue
    date = file_name[:8]
    # file = pd.read_csv(datapath + df.iloc[:, 0][0] + '/' + df.iloc[:, 0][0] + '-AAPL.csv')[['timeHMs', 'nrTrades']]
    file = pd.read_csv(datapath + file_name)[['timeHMs','nrTrades']]
    # file['date'] = date
    file['group'] = file.timeHMs//5
    groupped = file.groupby('group').sum()
    groupped.timeHMs = (groupped.timeHMs-10)//5
    file = groupped
    file.timeHMs = file.timeHMs.apply(lambda x:str(x).rjust(4,'0'))
    file.timeHMs = file.timeHMs.apply(lambda x:x[:2]+':'+x[2:])
    # date = df.iloc[:, 0][0]
    file['date'] = date
    file.date = file.date.apply(lambda x: str(x))
    file.date = file.date.apply(lambda x: x[:4]+'-'+x[4:6]+'-'+x[6:] + ' ')
    file['time'] = file.date + file.timeHMs
    file.time = file.time.apply(pd.to_datetime)
    df = file[['time', 'nrTrades']]
    df_lst.append(df)
    # file.nrTrades.plot()
    # plt.show()
dflst = pd.concat(df_lst)


dflst = dflst.reset_index();dflst = dflst.drop('group',axis = 1)

# dflst = dflst.drop('index',axis = 1)
dflst.to_csv('apple.csv')


df  = pd.read_csv('apple.csv')
df['hour'] = df.iloc[:,1].apply(lambda x:pd.to_datetime(x).time())
df1 = df.groupby('hour').mean()
# dflst.nrTrades.plot()d
# plt.show()
df1.nrTrades.plot()
plt.show()


# %%%%%%%%%%%%

import pandas as pd
from tqdm import tqdm
df = pd.read_csv("symbols.csv",index_col=0)
sym = df.iloc[:,0].apply(lambda x: x[9:-4])

for j in (13,90):
    stock = sym[j]
    print(f">>>>>>>>>>>>>>> stock {j} {sym[j]}")
    datapath = "/data/cholgpu01/not-backed-up/datasets/graf/data/Minutely_LOB_2017-19/"
    from os import listdir;import pandas as pd;import numpy as np
    dirs = [f for f in listdir(datapath)]
    dirs = sorted(dirs)
    df = pd.DataFrame(dirs)
    date = df.iloc[:, 0].values[0]
    file = pd.read_csv(datapath + date + '/' + date +'-'+stock+'.csv')[['timeHMs', 'nrTrades']]
    time_stamp = file.timeHMs.values
    df_lst = []

    for i in range(len(df)):
        date = df.iloc[:, 0].values[i]
        try:
            file = pd.read_csv(datapath + date + '/' + date + '-'+stock+'.csv')[['timeHMs', 'nrTrades']]
            file['date'] = date
            df_lst.append(file)
        except:
            pass

    dflst = pd.concat(df_lst)
    dflst = dflst.reset_index()
    dflst = dflst.drop('index',axis = 1)
    dflst.to_csv(datapath +'output/'+stock+'.csv')




