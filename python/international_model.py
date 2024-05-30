#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib
import statsmodels.api as sm
#from statsmodels.discrete import NegativeBinomial, Poisson

X = pd.read_csv("proc_data/ukraine_X.csv", index_col = 0)
ydf = pd.read_csv("proc_data/syn_y.csv", index_col = 0)
ydf.index = pd.to_datetime(ydf.index)

#tr = 'hump'
#tr = 'handle'
tr = 'all'

total_outflow = 'ols'
#total_outflow = 'nb'
#total_outflow = 'poisson'

hump_quad = True

hump_end = pd.to_datetime('2022-03-22')

if tr == 'hump':
    start_date = pd.to_datetime('2022-02-23')
    #start_test = pd.to_datetime('2022-04-16')
    start_test = pd.to_datetime('2022-03-22')
elif tr == 'all':
    start_date = pd.to_datetime('2022-02-23')
    start_test = pd.to_datetime('2022-08-01')
elif tr == 'handle':
    start_date = pd.to_datetime('2022-05-01')
    start_test = pd.to_datetime('2022-08-01')

keep_dates = start_date < ydf.index
ydf = ydf.loc[keep_dates,:]
train_dates = start_test > ydf.index

ydf_train = ydf.loc[train_dates,:]
ydf_test = ydf.loc[~train_dates,:]

## Moving average of Xs
ml = 15
mw = 5
llrs = np.zeros([ml,mw])
ooss = np.zeros([ml,mw])
for wi, w in enumerate(range(1,mw+1)):
    for li, lag in enumerate(range(ml)):
        if w > li:
            llrs[li,wi] = -np.inf
            ooss[li,wi] = -np.inf
        else:
            _, _, _, _, llf, _, _, _, nmse = lag_n_reg(lag, w)
                                        lag_n_reg(lag, w, start_test, ydf, Xu)
            llrs[li,wi] = llf
            ooss[li,wi] = nmse

fig = plt.figure(figsize=[12,6])
llf_lagi, llf_wi = np.unravel_index(np.argmax(llrs), llrs.shape)
#llrs[llf_lagi,llf_wi] = np.inf

plt.subplot(1,2,1)
plt.imshow(llrs)
plt.ylabel("Days back to look")
plt.xlabel("Days to average over")
plt.text(llf_wi,llf_lagi,s='X', color = 'red')
plt.colorbar()
plt.title("Model Fit")

oos_lagi, oos_wi = np.unravel_index(np.argmax(ooss), llrs.shape)
#ooss[oos_lagi,oos_wi] = np.inf
plt.subplot(1,2,2)
plt.imshow(ooss)
plt.ylabel("Days back to look")
plt.xlabel("Days to average over")
plt.text(oos_wi,oos_lagi,s='X', color = 'red')
plt.colorbar()
plt.title("Prediction Accuracy")

plt.tight_layout()
plt.savefig("surface.pdf")
plt.close()

Xsm, ysm, fit, fitd, llf, Xsm_test, ysm_test, pred, nmse = lag_n_reg(llf_lagi, llf_wi+1)
#Xsm, ysm, fit, fitd, llf, Xsm_test, ysm_test, pred, nmse = lag_n_reg(oos_lagi, oos_wi+1)

fit.summary()

fig = plt.figure()
#plt.plot(ysm)
plt.subplot(2,1,1)
plt.plot(ysm, label = 'Data')
plt.plot(np.exp(fitd), label = 'Fit')
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=8)
plt.title("In Sample")
plt.legend()

plt.subplot(2,1,2)
plt.plot(ysm_test, label = 'Data')
plt.plot(pred, label = 'Predicted')
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=8)
plt.title("Out of Sample")
plt.legend()

plt.tight_layout()
plt.savefig("ydf.pdf")
plt.close()
