import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings;

warnings.simplefilter("ignore", category=FutureWarning)
from os import listdir;
from os.path import isfile, join
from r_output import Config
from sklearn.metrics import r2_score

pd.set_option('display.max_columns', None)


def platform():
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    import warnings;
    warnings.simplefilter("ignore", category=FutureWarning)
    from os import listdir;
    from os.path import isfile, join
    from data import Config

    pd.set_option('display.max_columns', None)

    import platform  # Check the system platform
    if platform.system() == 'Darwin':
        print("Running on MacOS")
        data_path = "/r_output/04_r_output_raw_data_10/"
    elif platform.system() == 'Linux':
        print("Running on Linux")
        raise NotImplementedError
    else:
        print("Unknown operating system")

    onlyfiles = sorted([f for f in listdir(data_path) if isfile(join(data_path, f))])
    return data_path, onlyfiles

data_path, onlyfiles = platform()

for i in range(len(onlyfiles)):
    df = pd.read_csv(data_path + onlyfiles[i])
