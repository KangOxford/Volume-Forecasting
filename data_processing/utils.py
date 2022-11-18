# -*- coding: utf-8 -*-


def timestamp_format(message):
    message.time *= 1000000000
    message.time = message.time.astype("datetime64[ns]")
    message = message.reset_index()
    message = message.set_index('time')
    message = message.drop(['index','level_0'],axis = 1)
    return message
    
def plot_single_value(value):
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    figure(figsize=(50, 10), dpi=80)
    plt.plot(value)