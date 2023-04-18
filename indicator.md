| Type | Column Name | Description |
| --- | --- | --- |
| basic | symbol | Symbol of the Stock |
| basic | date | Date of the observation |
| basic | timeHMs | Start time of the interval in hours, minutes, and seconds |
| basic | timeHMe | End time of the interval in hours, minutes, and seconds |
| basic | intrSn | The intraday interval: open(9:30-10:00), mid, close(15:30-16:00) |
| basic | ntn | Traded notional in the bucket |
| basic | volBuyNotional | The buy orders in notional |
| basic | volSellNotional | The sell orders in notional |
| basic | nrTrades | Number of trades executed |
| basic | volBuyNrTrades_lit | Number of buy trades executed for limit orders |
| basic | volSellNrTrades_lit | Number of sell trades executed for limit orders |
| basic | volBuyQty | Total quantity for the buy order |
| basic | volSellQty | Total quantity for the sell order |
| daily | daily_ntn | Daily net traded notional |
| daily | daily_volBuyNotional | Daily buy orders in notional |
| daily_aggregate | daily_volSellNotional | Daily sell orders in notional |
| daily | daily_nrTrades | Daily sell orders in notional |
| daily | daily_volBuyNrTrades_lit | Daily number of buy trades executed for limit orders |
| daily | daily_volSellNrTrades_lit | Daily number of sell trades executed for limit orders |
| daily | daily_volBuyQty | Daily total quantity bought |
| daily | daily_volSellQty | Daily total quantity sold |
| daily | daily_qty | Daily total quantity traded |
| intraday | intraday_ntn | Intraday net traded notional |
| intraday | intraday_volBuyNotional | Intraday buy orders in notional |
| intraday | intraday_volSellNotional | Intraday sell orders in notional |
| intraday | intraday_nrTrades | Number of trades executed during the intraday session |
| intraday | intraday_volBuyNrTrades_lit | Number of buy trades executed limit orders during intraday |
| intraday | intraday_volSellNrTrades_lit | Number of sell trades executed limit orders during intraday |
| intraday | intraday_volBuyQty | Total quantity bought during intraday |
| intraday | intraday_volSellQty | Total quantity sold during intraday |
| intraday | intraday_qty | Total quantity traded during intraday |
| task | qty | Total quantity traded during the bucket |
| task | VO | The target we want to forecast(qty_lag1) |
