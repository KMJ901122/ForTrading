from __future__ import absolute_import
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def generate_point(data, positions):
    x_short=[]
    x_long=[]
    for x in positions:
        if x[0]==-1.0:
            x_short.append(x[1])
        else:
            x_long.append(x[1])

    y_long=[data[t] for t in x_long]
    y_short=[data[t] for t in x_short]
    return x_short, y_short, x_long, y_long

def plot_point(data_1, profit, x_short, y_short, x_long, y_long, title='Samsung'):
    fig, ax=plt.subplots(1, 1, figsize=(12, 12))
    plt.xlim([0, len(data_1)-1])
    plt.scatter(x_long, y_long, c='blue', s=100)
    plt.scatter(x_short, y_short, c='red', s=100)
    plt.plot(data_1)
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title(title)
    buy_and_hold=int(data_1[-1]-data_1[0])
    textstr='Profit: {}\nBuy and Hold: {}'.format(profit, buy_and_hold)
    textbox = offsetbox.AnchoredText(textstr, loc=1)
    ax.add_artist(textbox)

    # plt.text('Buy and hold: {}'.format(data_1[-1]-data_1[0]))
    # plt.figure(figsize=(15, 15))
    # plt.xlim([0, len(data_2)-1])
    # plt.scatter(x_long_2, y_long_2, c='blue', s=100)
    # plt.scatter(x_short_2, y_short_2, c='red', s=100)
    # plt.subplot(212)
    # plt.plot(data_2)
    plt.show()

'''
Usage example
plot_point(data, 'Samsung', *generate_point(data, positions))
'''

def plot_compare(step_list, list, name=None):
    if name==None:
        raise ValueError("name is None! Determine whether name is loss or reward")
    for i in range(len(list)):
        plt.plot(step_list, list[i][0], label=list[i][1])

    plt.ylabel(name)
    plt.xlabel('Step')
    plt.legend(loc='upper left')
    plt.show()
