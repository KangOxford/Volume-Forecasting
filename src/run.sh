export PYTHONPATH=$PYTHONPATH:/Users/kang/CMEM/
/Users/kang/opt/anaconda3/envs/x/bin/python /Users/kang/CMEM/data/01.data_prepare.py
/Users/kang/opt/anaconda3/envs/x/bin/python /Users/kang/CMEM/data/02.data_convert_before_r.py
/usr/local/bin/R --slave --no-save --no-restore --no-site-file --no-environ -f /Users/kang/CMEM/run.r --args ""
/Users/kang/opt/anaconda3/envs/x/bin/python /Users/kang/CMEM/r_output/03.transform.py

