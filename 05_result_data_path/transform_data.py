import pandas as pd
from os import listdir;from os.path import isfile, join
import re

data1 = "/Users/kang/Volume-Forecasting/05_result_data_path/Lasso/"
onlyfiles = sorted([f for f in listdir(data1) if isfile(join(data1, f))])

if __name__=="__main__":
    for item in onlyfiles:
        with open(data1+item) as f:
            lines = f.readlines()
        print(lines)
        # Assuming "lines" is a list containing the string values
        result = []
        for line in lines:
            # Use regex to extract values and convert to appropriate type
            values = []
            for v in re.findall(r"'([^']*)'", line):
                if v.isdigit():
                    values.append(int(v))
                elif re.match(r'^-?\d+(?:\.\d+)$', v):
                    values.append(float(v))
                else:
                    values.append(v)
            result.append(values)
            # Split values into sublists of length 3
            while len(values) > 3:
                result.append(values[:3])
                values = values[3:]
            # Append remaining values to result
            result.append(values)

        # Remove empty lists from result
        result = [r for r in result if r]

        new_result = [[result[i][j], result[i][j + 1], result[i][j + 2]] for i in range(0, len(result), 3) for j in
                      range(0, 3)]

        # new_r = []
        # for value in result:
        #     while len(values) > 3:
        #         new_r.append(values[:3])
        #         values = values[3:]
        #     # Append remaining values to result
        #     new_r.append(values)


        # result should be a nested list with each list inside has a len of 3,
        # you need to delete [] and for list similar to
        # ['ABBV', 20171228, 1171.0997814492644, 'ABC', 20170720, 4382.204771859686],
        # ['ABBV', 20171228, 1171.0997814492644, 'ABC', 20170720, 4382.204771859686, 'ABCD', 20170720, 4382.204771859686],
        # ['ABBV', 20171228, 1171.0997814492644, 'ABC', 20170720, 4382.204771859686, 'ABCD', 20170720, 4382.204771859686, 'ABCDE', 20170720, 4382.204771859686],
        #  you need to split it into two
pattern1 = r'[A-Z]'
pattern2 = r'^\d+$'
pattern3 = r'\d+\.\d+'
lsts = []
for line in lines:
    lst = []
    for v in line.split("\'"):
        match1 = re.match(pattern1, v)
        match2 = re.match(pattern2, v)
        match3 = re.match(pattern3, v)
        # print(v)
        if match1:
            lst.append(v)
        if match2:
            lst.append(v)
        if match3:
            lst.append(v)
    lsts.append(lst)
newlsts = []
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
newdf = pd.DataFrame(newlsts, columns=['symbol','date','value'])
newdf.to_csv("")

# digits = []
# for item in lsts:
#     digits.append(len(item))
# set(digits)
