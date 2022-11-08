# Volume-Transformer
Trading Volume Time Series Prediction in LOB


# Timeline:
## Week 1 Feature Engineering
* Cross auction Implemention
* [Time_Series_Transformer](https://huggingface.co/docs/transformers/model_doc/time_series_transformer)
* Clocks types
  * calender clock
  * volume clock
  * tick clock
* The timewindow size: 1m(default), 1s, 10s, 30s [intra day]
* Feature Engineering
  * cancellation is important
  * top k(default 3) largest 
  * aggressive buy and sell, buy MO
  * 
