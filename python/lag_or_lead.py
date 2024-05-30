#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib
from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson
import statsmodels.api as sm
from tqdm import tqdm
import matplotlib.dates as mdates
from adjustText import adjust_text

exec(open("python/lib.py").read())

X = pd.read_csv("proc_data/ukraine_X.csv", index_col = 0)
ydf = pd.read_csv("proc_data/syn_y.csv", index_col = 0)
ydf.index = pd.to_datetime(ydf.index)
X.index = pd.to_datetime(X.index)

#ydf = ydf.drop(['Hungary','Poland','Slovakia'], axis = 1)

start_test = pd.to_datetime('2022-08-01')
end_test = pd.to_datetime('2022-09-03')

start_test = pd.to_datetime('2022-08-01')

groups = list(set([x.split('-')[0] for x in X.columns]))
vpg = dict([(i,[x for x in X.columns if x.split('-')[0]==i]) for i in groups])

group2source2 = dict(group2source)
group2source2['trends']='Google Trends'
group2source2['news']='Newspapers'

#for g in ['gdelt']:
#for country in countries:
for country in ['hps_outflow']:
    lw = pd.DataFrame(np.zeros([len(X.columns),3]))
    lw.index = X.columns
    lw.columns = ['Lag','Window','R2']
    for vi,v in enumerate(tqdm(X.columns)):
        Xc = pd.DataFrame(X.loc[:,v])
        costname = v+"_cost.pdf"
        lag, w, fit, nmae_oos, nmae_is = est_lag(ml, mw, end_test, ydf, Xc, hump_quad = False, pos_lag = False, fname = costname, pred_col = country)

        lw.iloc[vi,0] = lag
        lw.iloc[vi,1] = w
        lw.iloc[vi,2] = fit.rsquared

    jitr = np.zeros(len(X.columns))
    yvar = 'R2'

    fig = plt.figure(figsize=[10,5])
    for xi,xvar in enumerate(['Lag','Window']):
        plt.subplot(1,2,1+xi)
        cols = {'trends':'red','buzz':'green','news':'orange','events':'purple','gdelt':'gray'}
        cc = [cols[x.split('-')[0]] for x in X.columns]
        plt.scatter(lw.loc[:,xvar], lw.loc[:,yvar]+jitr, color = cc, s = 75)
        texts = []
        for vi,v in enumerate(X.columns):
            g = v.split('-')[0]
            t = v.split('-')[1].title()
            texts.append(plt.text(lw.loc[v,xvar], lw.loc[v,yvar]+jitr[vi], t, fontdict={'size':8}))
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='blue', lw=2,alpha=0))
        plt.title("Best Fitting "+xvar)#+" by Variable ("+country+")")
        ax = plt.gca()
        if xvar=='Lag':
            ax.set_xlabel("Lag (Days)")
        elif xvar=='Window':
            ax.set_xlabel("Window (Days)")
        ax.set_ylabel("Fit (R squared)")

        if xvar=='Lag':
            #ax = ax.gca()
            minr2 = np.min(lw.loc[:,'R2'])
            maxr2 = np.max(lw.loc[:,'R2'])
            alpha = 0.1
            ax.fill_between([0,ml], minr2, maxr2, alpha = alpha, color = 'blue')
            ax.fill_between([0,-ml], minr2, maxr2, alpha = alpha, color = 'red')
            ylevel = np.quantile(lw.loc[:,'R2'], 0.00)*1.4
            ax.text(ml/2, ylevel, 'Leading', color = 'blue', horizontalalignment='center')
            ax.text(-ml/2, ylevel, 'Lagging', color = 'Red', horizontalalignment='center')
    
    plt.savefig("images/"+country+"_wlag_fit.pdf")
    plt.close()

    ### Make legend
    fig = plt.figure(figsize=[1.8,1.5])
    ax = plt.gca()
    sc = []
    for v in cols:
        sc.append(ax.scatter(np.mean(lw.loc[:,xvar]), np.mean(lw.loc[:,yvar]), label = group2source2[v], color = cols[v]))
    ax.legend()
    for ss in sc:
        ss.set_alpha(0)
    plt.axis('off')
    plt.savefig("images/wlag_legend.pdf")
    plt.close()

