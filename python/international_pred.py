#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import matplotlib.dates as mdates
from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson
import statsmodels.api as sm
from tqdm import tqdm
import scipy.stats as ss
from time import time

tt = time()

exec(open("python/lib.py").read())
test_every = 7

X = pd.read_csv("proc_data/ukraine_X.csv", index_col = 0)
ydf = pd.read_csv("proc_data/syn_y.csv", index_col = 0)
ydf.index = pd.to_datetime(ydf.index)
X.index = pd.to_datetime(X.index)

notenvr = [x for x in X.columns if 'environment' not in x]
X = X.loc[:,notenvr]

#tr = 'handle'
tr = 'all'

model = 'ols'
#model = 'nb'
#model = 'poisson'

start_date = pd.to_datetime('2022-02-23')
test_window_start = pd.to_datetime('2022-02-24') + pd.Timedelta(days=14)
last_pred_day = np.min([np.max(ydf.index), np.max(X.index)])
hump_end = pd.to_datetime('2022-03-22')

keep_dates = start_date < ydf.index

ydf = ydf.loc[keep_dates,:]

test_starts = [pd.to_datetime(x) for x in ['2022-03-10','2022-04-10','2022-05-10','2022-06-10']]
test_ends = [x+pd.Timedelta(days=21) for x in test_starts]

for dt in ['single']:
    for i in range(10):
        print(dt)

    ## Moving average of Xs

    if dt == 'single':
        datasets = list(X.columns)
    if dt == 'group':
        datasets = list(set([x.split('-')[0] for x in X.columns]))
    if dt == 'pair':
        ds = list(set([x.split('-')[0] for x in X.columns]))
        datasets = []
        for ds1 in ds:
            for ds2 in ds:
                datasets.append(ds1+'-'+ds2)

    D = len(datasets)

    pred_err = np.zeros([D,len(test_starts)])
    pred_err = pd.DataFrame(pred_err)
    pred_err.index = datasets
    pred_err.columns = test_starts

    lag_est = np.zeros([D,len(test_starts)])
    lag_est = pd.DataFrame(lag_est)
    lag_est.index = datasets
    lag_est.columns = test_starts

    window_est = np.zeros([D,len(test_starts)])
    window_est = pd.DataFrame(window_est)
    window_est.index = datasets
    window_est.columns = test_starts

    null_err = pd.Series(np.zeros(len(test_starts)))
    null_err.index = test_starts

    #for country in countries:
    for country in ['hps_outflow']:
        for di,ds in enumerate(tqdm(datasets)):
            if ds =='allstar':
                Xu = X.loc[:,['trends-physical','trends-travel','buzz-health','trends-food']]
            elif ds == 'pcs':
                aa = pd.Categorical([x.split('-')[0] for x in X.columns])
                b = aa.unique().to_numpy().astype(str)
                a = aa.codes
                is_diff = np.diff(a)!=0
                is_diff = np.concatenate([[False],is_diff])
                breaks = np.arange(len(X.columns))[is_diff]
                breaks = np.arange(len(X.columns))[is_diff]
                aa = np.concatenate([[0],breaks,[X.shape[1]+3]])
                pcs = pd.DataFrame(np.zeros([X.shape[0],len(b)]))
                pcs.index = X.index
                pcs.columns = b
                Xc = (X - np.mean(X,axis=0)[np.newaxis,:]) / np.std(X,axis=0)[np.newaxis,:]
                for i in range(len(aa)-1):
                    submat = Xc.iloc[:,aa[i]:aa[i+1]]
                    V = np.transpose(np.linalg.svd(submat, full_matrices = False)[2])
                    v = V[:,0]
                    z = submat @ v
                    pcs.iloc[:,i] = z
                Xu = pcs
            else:
                if dt == 'single':
                    Xu = X.loc[:,[ds]]
                if dt == 'group':
                    tc = [x for x in X.columns if x.split('-')[0]==ds]
                    Xu = X.loc[:,tc]
                if dt == 'pair':
                    tc1 = [x for x in X.columns if x.split('-')[0]==ds.split('-')[0]]
                    tc2 = [x for x in X.columns if x.split('-')[0]==ds.split('-')[1]]
                    tc = list(set(tc1+tc2))
                    Xu = X.loc[:,tc]

            for si,start_test in enumerate(tqdm(test_starts)):
                # For pair, skip the first few periods cause performance is so bad.
                if dt=='pair' and si < 3:
                    pred_err.iloc[di,si] = np.nan
                    lag_est.iloc[di,si] = np.nan
                    window_est.iloc[di,si] = np.nan
                else:
                    prefix = ds+str(start_test).split(' ')[0]
                    fn = "images/surface"+prefix+".pdf"
                    opt_lag, opt_w, fit, nmae_oos, nmae_is = est_lag(ml, mw, start_test, ydf, Xu, hump_quad = False, fname=fn, pred_col = country, test_end = test_ends[si])

                    pred_err.iloc[di,si] = -nmae_oos
                    lag_est.iloc[di,si] = opt_lag
                    window_est.iloc[di,si] = opt_w

                # This is being needlessly recomputed, but is cheap.
                end_test = pd.to_datetime('2022-09-01')
                sd = start_test - pd.Timedelta(days=7)
                gdy = np.logical_and(ydf.index >= sd, ydf.index<start_test)
                gdy_test = np.logical_and(ydf.index>=start_test, ydf.index<=end_test)
                mm = np.mean(ydf.loc[gdy,country])
                null_last = -np.mean(np.abs(ydf.loc[gdy_test,country]-mm))
                null_err[si] = null_last

        if dt in ['single','pair']:
            K = 10
        else:
            K = len(datasets)

        pe = pred_err.copy()
        pe.loc['Prev Week Mean',:] = -null_err
        #ranks = pe.apply(ss.rankdata)
        #ranks = np.mean(ranks, axis = 1)
        #kinds = np.argpartition(-np.array(ranks), -K)[-K:]
        #best_K = list(ranks[kinds].index)


        #pe.index[mr.iloc[:,1]]
        #pe.index[best_K]

        #ranks = np.mean(ranks, axis = 1)
        #kinds = np.argpartition(-np.array(ranks), -K)[-K:]
        #best_K = list(ranks[kinds].index)

        #cols = ['red','blue','green','orange','cyan','gold','maroon','violet','pink','olive']
        #assert len(cols) >= K_use

        #yplot = 'raw'
        yplot = 'norm'
        
        if yplot == 'norm':
            pe = np.log10(pe)
            pe = (pe - np.mean(pe)[np.newaxis,:]) / np.std(pe)[np.newaxis,:]

        pec = pe.loc[:,test_starts[0:1]]
        pec[test_starts[1]] = np.mean(pe.loc[:,test_starts[1:]], axis = 1)

        K_per = K//pec.shape[1]
        ranks = pec.apply(ss.rankdata)
        mr = ranks.apply(lambda x: np.argpartition(-x, -K_per)[-K_per:])
        best_K = list(set(np.array(mr).flatten()))
        K_use =len(best_K)
        print(best_K)

        cm = mpl.cm.get_cmap('tab20')
        cols = [cm(i/K_use) for i in range(K_use)]

        #fig = plt.figure(figsize=[7.5,5])
        #fig = plt.figure(figsize=[7.5,3.5])
        #fig = plt.figure(figsize=[10,3.5])
        fig = plt.figure(figsize=[10,4])

        plt.subplot(1,3,1)
        plt.plot(ydf[country][ydf.index<=np.max(test_ends)], color = 'gray')
        plt.vlines(test_starts, np.min(ydf[country]), np.max(ydf[country]), color = ['black','red','red','red'], linestyle='--')

        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.set_xticks(test_starts)
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        if country=='hps_outflow':
            plt.title("Slovakia + Hungary + Poland")
        else:
            plt.title(country)
        plt.ylabel("Individuals")

        for ppi in range(2):
            plt.subplot(1,3,2+ppi)
            pps = []
            for ii,i in enumerate(pec.index):
                #print(best_K)
                if ii in best_K:
                    if i == 'Prev Week Mean':
                        lab = i
                        color = 'black'
                    else:
                        # TODO: This code block should be a function.
                        pre = group2source[i.split('-')[0]]
                        t = pre +'-'+ i.split('-')[1].title()
                        lab = t
                        bind = np.where(np.array(best_K)==ii)[0][0]
                        color = cols[bind]
                    w = 1
                    if bind <= 10:
                        linestyle = '--'
                    else:
                        linestyle = '-'
                    alpha = 1.0
                else:
                    lab = None
                    color = 'gray'
                    linestyle = '-'
                    alpha = 0.0
                    w = 1
                
                pps.extend(plt.plot(pec.loc[i,:], label = lab, color = color, linestyle=linestyle, alpha = alpha, linewidth = w))
                if ii in best_K:
                    dat = pec.loc[i,:]
                    #plt.scatter(dat.index, np.array(dat), color = color, alpha = alpha, marker = '*', s = w*30)
                    pps.append(plt.scatter(dat.index, np.array(dat), color = color, alpha = alpha, s=w*30*2))
            
            if ppi==1:
                plt.legend(prop={'size':13})
                plt.axis("off")
                for pscat in pps:
                    pscat.set_alpha(0)
            elif ppi==0:
                #plt.xlabel("Testing starts...")
                if yplot == 'raw':
                    plt.ylabel("Individuals prediction was off by")
                    plt.yscale("log")
                else:
                    plt.ylabel("Normalized Prediction Error")

                plt.title("Three Week 'Prediction Error'")

                ax = plt.gca()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.set_xticks(test_starts[:2], labels = ['Early','Late'])
                
                spe = pec.iloc[best_K,:]
                ax.set_ylim(spe.min().min()-0.3, spe.max().max()+0.3)
                ##

        fname = country+"_"+dt+"_pred_over_time.pdf"# if pi==0 else country+"_"+dt+"_pred_over_time_abr.pdf"
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

        fig = plt.figure()
        jitsd = 1e-1
        for ii,i in enumerate(pred_err.index):
            if i in best_K:
                lab = i
                bind = np.where(np.array(best_K)==i)[0][0]
                color = cols[bind]
                alpha = 1.0
            else:
                lab = None
                color = 'gray'
                alpha = 0.5

            plt.plot(lag_est.loc[i,:]+np.random.normal(scale=jitsd,size=lag_est.shape[1]), label = lab, color = color)
            plt.plot(window_est.loc[i,:]+np.random.normal(scale=jitsd,size=lag_est.shape[1]), color = color, linestyle='--')
        plt.legend()
        plt.title("Lag Estimates by Source")
        #plt.xlabel("Testing starts...")
        plt.ylabel("Estimate (jittered)")
        plt.savefig('images/'+country+"_"+dt+"_lagests.pdf")
        plt.close()

tt1 = time()
print("Execution time")
print(tt1-tt)
