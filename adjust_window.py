#!/usr/bin/env python3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import deque

def read_data():
    dicts = []
    with open('data/rolling_avgs.txt') as fh:
        dicts = [json.loads(line.strip()) for line in fh]

    if not dicts:
        print("Couldn't load data")
        quit()

    raw = []
    for d in dicts:
        ep = d['episode']
        # reward, time, rolling average of past 100 rewards
        ep_data = ep['r'], ep['t']
        ep_data = ep['r'], ep['t']
        raw.append(ep_data)

    raw = np.array(raw)
    return raw

def rolling_mean(data, window_size, cols=None):
    """Applies as rolling mean to the input and stores the result in data['a']
    
    By default, all columns are averaged. Otherwise, averaging is applied only
    to the given columns, and the remaining columns are returned unchanged.
    """
    if not '__iter__' in dir(cols):
        cols = (cols,)
    window = deque((), maxlen=window_size)
    means = []
    raw = []
    for row in data:
        window.append(row)
        if len(window) == window_size:
            array = np.array(window)
            mean = array.mean(axis=0)
            means.append(mean)
            raw.append(window[-1])
    means = np.array(means)
    raw = np.array(raw)
    m_cols = means.T
    r_cols = raw.T
    final = []
    for col in range(len(m_cols)):
        if col in cols:
            final.append(m_cols[col])
        else:
            final.append(r_cols[col])
    return np.array(final).T

data = read_data()
avgs = rolling_mean(data, 1000, 0)

while True:
    plt.plot(avgs[:,1], avgs[:,0])
    plt.show()
    response = input("Continue? [Y/n] ")
    if response:
        response = response.lower()
        if response[0] == 'n':
            break
