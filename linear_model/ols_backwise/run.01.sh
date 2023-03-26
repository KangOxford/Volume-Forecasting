#!/bin/bash

#cd /Users/kang/Volume-Forecasting/linear_model/ols_backwise/01_data_prepare.py

# Run the first Python file
python /Users/kang/Volume-Forecasting/linear_model/ols_backwise/01_data_prepare.py

# Run the second Python file
python /Users/kang/Volume-Forecasting/linear_model/ols_backwise/02.1_data_prepare_intr_daily_groupby.py

# Run the third Python file
python /Users/kang/Volume-Forecasting/linear_model/ols_backwise/10.1_ols_mv_15min_fs.py

python /Users/kang/Volume-Forecasting/linear_model/ols_backwise/11.1_ols_mv_15min_numba.py

python /Users/kang/Volume-Forecasting/linear_model/ols_backwise/12.1_ols_mv_15min_test.py
