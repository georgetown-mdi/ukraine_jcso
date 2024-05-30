#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## This script generates synthetic border crossings, with a quadratic hump at the beginning and a constant later.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(123)

date_start = pd.to_datetime('2022-02-24')
hump_end = pd.to_datetime('2022-04-01')
date_end = pd.to_datetime('2022-10-18')

hump_rad = (hump_end - date_start) / 2
hump_mid = date_start + hump_rad 

mig_max = 150000
mig_start = 0
mig_steady = 10000

t = pd.date_range(date_start, date_end)

is_hump = t <= hump_end

mu_hump = mig_max - (mig_max-mig_steady) * np.square((t-hump_mid).days) / np.square(hump_rad.days)
mu_steady = mig_steady
mu_y = mu_hump*is_hump + mu_steady*(~is_hump)

y = np.random.normal(loc=mu_y,scale=np.sqrt(mu_y)*10)

fig = plt.figure()
plt.plot(t,y)
plt.savefig("syndata.pdf")
plt.close()

df = pd.DataFrame(y)
df.index = t
df.columns = ['hps_outflow']

dow = df.index.weekday
dowdf = pd.get_dummies(dow, drop_first = True)
dowdf.columns = ['T','W','Th','F','S','Su']
dowdf.index = df.index
df = pd.concat([df, dowdf], axis = 1)

df.to_csv("proc_data/syn_y.csv")
