import pandas as pd
from os import listdir;from os.path import isfile, join
import re

data1 = "/Users/kang/Volume-Forecasting/05_result_data_path/Lasso/"
onlyfiles = sorted([f for f in listdir(data1) if isfile(join(data1, f))])

if __name__=="__main__":
    for file in onlyfiles:
        with open(data1+file) as f:
            lines = f.readlines()
        ify = len(lines[0].split(",")) == 5
        pattern1 = r'[A-Z]'
        pattern2 = r'^\d+$'
        # pattern3 = r'\d+\.\d+'
        pattern3 = r'-?\d+\.\d+'
        # if ify: pattern4 = r'-?\d+\.\d+'
        lsts = []
        for line in lines:
            lst = []
            for v in line.split("\'"):
                match1 = re.match(pattern1, v)
                match2 = re.match(pattern2, v)
                match3 = re.match(pattern3, v)
                # if ify: match4 = re.match(pattern4, v)
                # print(v)
                if match1:
                    lst.append(v)
                if match2:
                    lst.append(v)
                if match3:
                    lst.append(v)
                # if ify and match4:
                #     lst.append(v)
            lsts.append(lst)
        digits = []
        for item in lsts:
            digits.append(len(item))
        num = set(digits)

        newlsts = []
        if ify:
            for item in lsts:
                if len(item) == 0:
                    pass
                elif len(item) == 4:
                    newlsts.append(item)
                elif len(item) == 8:
                    newlsts.append(item[:4])
                    newlsts.append(item[4:])
                else:
                    raise NotImplementedError
        else:
            for item in lsts:
                if len(item) == 0:
                    pass
                elif len(item) == 3:
                    newlsts.append(item)
                elif len(item) == 6:
                    newlsts.append(item[:3])
                    newlsts.append(item[3:])
                else:
                    raise NotImplementedError

        for item in newlsts:
            item[1] = int(item[1])
            item[2] = float(item[2])
            if ify:item[3] = float(item[3])
        if ify:
            newdf = pd.DataFrame(newlsts, columns=['symbol', 'date', 'true','pred'])
        else:
            newdf = pd.DataFrame(newlsts, columns=['symbol', 'date', 'value'])
            newdf.to_csv(data1 + file)


# digits = []
# for item in lsts:
#     digits.append(len(item))
# set(digits)
