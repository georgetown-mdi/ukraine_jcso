#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
from statsmodels.stats.outliers_influence import variance_inflation_factor

exec(open("python/lib.py").read())

X = pd.read_csv("proc_data/ukraine_X.csv", index_col = 0)

P = X.shape[1]

## Correlation between predictors.
Xc = (X - np.mean(X,axis=0)[np.newaxis,:]) / np.std(X,axis=0)[np.newaxis,:]
corr = (Xc.T @ Xc) / Xc.shape[0]
vifs = [variance_inflation_factor(Xc, i) for i in range(Xc.shape[1])]
fig = plt.figure(figsize=[8,6])
plt.imshow(corr, cmap = plt.get_cmap('RdBu'), vmin = -1, vmax = 1)
#for p in range(len(vifs)):
#    plt.text(p,p,np.round(vifs[p],1), ha="center", va="center", color="orange", weight = 'bold')
ax = plt.gca()
ax.set_yticks(np.arange(len(Xc.columns)))
ax.set_yticklabels([x.split('-')[1].title() for x in Xc.columns])
#plt.yticks(weight='bold')
xl = ax.get_xlim()
yl = ax.get_ylim()

##### Show grouping of variables
aa = pd.Categorical([x.split('-')[0] for x in X.columns])
b = aa.unique().to_numpy().astype(str)
a = aa.codes
is_diff = np.diff(a)!=0
is_diff = np.concatenate([[False],is_diff])
breaks = np.arange(len(X.columns))[is_diff]
for l in breaks:
    plt.vlines(l-0.5,-0.5,X.shape[1]+0.5, color = 'black',linewidth=2)
    plt.hlines(l-0.5,-0.5,X.shape[1]+0.5, color = 'black',linewidth=2)
 
aa = np.concatenate([[0],breaks,[X.shape[1]+3]])
mids = np.convolve(aa,[1/2,1/2],mode='valid')
mids -= 0.5
mids[-1] -= 1
mids[-2] -= 0.5
ax.set_xticks(mids)
#ax.set_xticklabels(b)
ax.set_xticklabels([group2source[bi] for bi in b])
plt.xticks(weight='bold')
plt.yticks(weight='bold')
#plt.xlim(0,Xc.shape[1])

plt.title("Correlation between Predictors", weight = 'bold')

ax.set_xlim(xl)
ax.set_ylim(yl)

plt.colorbar()
plt.savefig("images/X_corr.pdf")
plt.close()

