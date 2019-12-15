#!/usr/bin/env python3
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json

def read_avgs():
    dicts = []
    with open('data/rolling_avgs.txt') as fh:
        dicts = [json.loads(line.strip()) for line in fh]

    if not dicts:
        print("Couldn't load data")
        quit()

    avgs = []
    for d in dicts:
        ep = d['episode']
        if not 'a' in ep:
            continue
        # reward, time, rolling average of past 100 rewards
        ep_data = ep['r'], ep['t'], ep['a']
        avgs.append(ep_data)

    avgs = np.array(avgs)
    return avgs

while True:
    avgs = read_avgs()
    plt.plot(avgs[:,1], avgs[:,2])
    plt.show()
    response = input("Continue? [Y/n] ")
    if response:
        response = response.lower()
        if response[0] == 'n':
            break
