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

exec(open("python/lib.py").read())

X = pd.read_csv("proc_data/ukraine_X.csv", index_col = 0)
ydf = pd.read_csv("proc_data/syn_y.csv", index_col = 0)
ydf.index = pd.to_datetime(ydf.index)
X.index = pd.to_datetime(X.index)

start_test = pd.to_datetime('2022-08-01')
end_test = pd.to_datetime('2022-09-03')

groups = list(set([x.split('-')[0] for x in X.columns]))
vpg = dict([(i,[x for x in X.columns if x.split('-')[0]==i]) for i in groups])

## Visualize everything internationally.
for g in tqdm(vpg):
    vs = vpg[g]
    nrow = int(np.ceil(len(vs)/2))
    fig = plt.figure(figsize=[7,nrow*2])
    for vi,v in enumerate(vs):
        Xc = pd.DataFrame(X.loc[:,v])

        lag, w, fit, nmae_oos, nmae_is  = est_lag(ml, mw, end_test, ydf, Xc, hump_quad = False)

        #print(fit.summary())

        plt.subplot(nrow,2,vi+1)
        plt.plot(ydf.loc[:max(X.index),'hps_outflow'])
        plt.plot(np.exp(fit.fittedvalues))
        pre = group2source[v.split('-')[0]]
        t = pre +'-'+ v.split('-')[1].title()
        plt.title(t)
        #ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.tick_params(axis='x', which='major', labelsize=7)
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.tight_layout()
    plt.savefig("images/fit/"+g+"_fit.pdf")
    plt.close()

## Get top performer of each group
# for pred_col in [
#pred_col = 'hps_outflow'
#pred_col = 'Moldova'


epg = {}
#for pred_col in countries:
for pred_col in ['hps_outflow']:
    epg[pred_col] = {}
    for v in tqdm(X.columns):
        Xc = pd.DataFrame(X.loc[:,v])
        lag, w, fit, nmae_oos, nmae_is = est_lag(ml, mw, end_test, ydf, Xc, hump_quad = False, pred_col = pred_col)
        epg[pred_col][v] = nmae_is

groups = ['buzz', 'trends', 'news', 'gdelt', 'events']
tops = {}
#for c in countries:
for c in ['hps_outflow']:
    tops[c] = {}
    for g in groups:
        g_vars = [(x,epg[c][x]) for x in epg[c].keys() if x.split('-')[0]==g]
        ind_best = np.argmax([x[1] for x in g_vars])
        name_best = g_vars[ind_best][0]
        tops[c][g] = name_best

topvars = dict([(c, list(tops[c].values())) for c in tops])

# Top for each
ncol = 3
#fig = plt.figure(figsize=[7,ncol*2])
#for c in countries:
for c in ['hps_outflow']:
    fig = plt.figure(figsize=[9,5])
    #for vi,v in enumerate(['buzz-economic','buzz-environmental','trends-travel','events-FATALITIES','gdelt-totalNegTone','news-physical']):
    #for vi,v in enumerate(['buzz-economic','trends-travel','trends-physical','events-FATALITIES','gdelt-Negative Tone','news-physical']):
    for vi,v in enumerate(topvars[c]):
        Xc = pd.DataFrame(X.loc[:,v])

        #lag, w, nmae, fit = est_lag(ml, mw, start_test, ydf, Xc, hump_quad = False)
        lag, w, fit, nmae_oos, nmae_is = est_lag(ml, mw, end_test, ydf, Xc, hump_quad = False)

        print(fit.summary())

        #plt.subplot(ncol,2,vi+1)
        plt.subplot(2,ncol,vi+1)
        p1 = plt.plot(ydf.loc[:end_test,'hps_outflow'], label = 'Observed')
        p2 = plt.plot(np.exp(fit.fittedvalues), label = 'Fit', linestyle='--')

        pre = group2source[v.split('-')[0]]
        t = pre +'-'+ v.split('-')[1].title()

        plt.title(t)
        if vi == 0:
            #plt.ylabel("Individuals Leaving UK")
            if c =='hps_outflow':
                plt.ylabel("Border Crossings")
            else:
                plt.ylabel("Arriving in "+c)
            #plt.legend()
        #ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.tick_params(axis='x', which='major', labelsize=7)
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        
    
    plt.subplot(2,ncol,2*ncol)
    plt.legend(handles=p1+p2)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("images/"+c+"_top_fit.pdf")
    plt.close()

# Special focus plot for trends-travel
c = 'hps_outflow'
fig = plt.figure(figsize=[4,3])
v = 'trends-travel'
Xc = pd.DataFrame(X.loc[:,v])

lag, w, fit, nmae_oos, nmae_is = est_lag(ml, mw, end_test, ydf, Xc, hump_quad = False)

p1 = plt.plot(ydf.loc[:end_test,'hps_outflow'], label = 'Observed', color = 'C0')
p2 = plt.plot(np.exp(fit.fittedvalues), label = 'Fit', linestyle='--', color = 'C1')

pre = group2source[v.split('-')[0]]

plt.title('Search Index Matches Migration')
plt.ylabel("Border Crossings")
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.tick_params(axis='x', which='major', labelsize=7)
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.legend(handles=p1+p2)

plt.tight_layout()
plt.savefig("images/travel_trends_fit.pdf")
plt.close()
