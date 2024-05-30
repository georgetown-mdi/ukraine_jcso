#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

hump_end = pd.to_datetime('2022-03-22')

group2source = {
        'events' : 'ACLED',
        'buzz' : 'Twitter',
        'trends' : "Trends",
        "news" : "News",
        "gdelt" : "GDELT"
        }

countries = ['Hungary', 'Poland', 'Slovakia', 'Moldova', 'Romania','hps_outflow']

## Lag/window params
#ml = 5
#mw = 5
#ml = 15
#mw = 10
ml = 23
mw = 30

def complete_the_hypercube(df, cols, dates = [], fill_value = np.nan):
    """
    Fills in missing combinations of a dataframe.
    df - A dataframe with missing combinations of stuff in cols.
    cols - A list (order matters!) letting us know how df should be indexed
    dates - A list telling us which columns contain dates. We will look for gaps in the dates as well as filling in any other combinations.
    fill_value
    """
    df.index = pd.MultiIndex.from_frame(df.loc[:,cols])
    uv = {}
    for col in cols:
        uv[col] = list(set(dfo[col]))
    for d in dates:
        uv[d] =pd.date_range(np.min(uv[d]), np.max(uv[d])) 
    vlist = []
    for n in uv:
        vlist.append(uv[n])
    #vlist = [uv[n] for n in uv]
    mult = pd.MultiIndex.from_product(vlist, names=cols)
    return df.reindex(mult, fill_value = fill_value)

##### International model.
def add_T2Xs(Xs, hump_end):
    t = np.linspace(0,1,Xs.shape[0])
    t2 = np.square(t) # TODO: Orthogonalize?
    T = pd.DataFrame([t,t2]).T
    T.index = Xs.index
    T = T * (T.index <= hump_end).astype(float)[:,np.newaxis]
    itm = T.index <= hump_end
    T.loc[itm,:] = (T.loc[itm,:] - np.mean(T.loc[itm,:])[np.newaxis,:]) / np.std(T.loc[itm,:])[np.newaxis,:]
    T.columns = ['hump_lin','hump_quad']
    T = T.join(pd.Series(np.ones(T.shape[0])*itm, index = T.index, name = 'hump_const'))
    Xs = Xs.join(T)

    return Xs

def lag_n_reg(lag, w, start_test, ydf, Xu, pred_col = 'hps_outflow', hump_quad = False, lik = 'gaus', test_end = None, verbose = False):
    start_date = pd.to_datetime('2022-02-24')
    dow = ['T','W','Th','F','S','Su']
    # Train-test split 
    train_dates = start_test > ydf.index
    ydf_train = ydf.loc[train_dates,:]
    ydf_test = ydf.loc[~train_dates,:]

    # Moving average
    Xw = Xu.copy().apply(lambda x: np.convolve(x, np.ones(w)/w, mode = 'same'), axis = 0)

    if hump_quad:
        Xw = add_T2Xs(Xw, hump_end)
    mu_x = np.array(np.mean(Xw.loc[start_test > Xw.index,:], axis = 0))
    sig_x = np.array(np.std(Xw.loc[start_test > Xw.index,:], axis = 0))
    sig_x[sig_x<1e-5] = 1
    Xw = (Xw - mu_x[np.newaxis,:]) / (sig_x[np.newaxis,:])

    # Lag index
    Xlag = Xw.copy()
    Xlag.index = Xlag.index + pd.Timedelta(days=lag)

    # Join to hps_outflow
    Xy = Xlag.join(ydf_train, how = 'inner')
    #good_til = np.logical_not(np.any(np.isnan(Xy), axis = 1))
    #good_til = np.logical_not(np.any(np.isnan(Xy), axis = 1))
    #Xy = Xy.loc[good_til,:]
    Xs = Xy.loc[:,list(Xu.columns)+dow]
    #Xs = Xy.drop(pred_col, axis = 1)

    ## Fit model
    isconst = np.var(Xs, axis = 0) < 1e-8
    Xs = Xs.loc[:,~isconst]
    Xsm = sm.add_constant(Xs, has_constant = 'raise')
    ysm = Xy[pred_col]

    if lik == 'nb':
        fit = NegativeBinomial(ysm, Xsm).fit()
    elif lik == 'poisson':
        fit = Poisson(ysm, Xsm).fit()
    elif lik == 'gaus':
        fit = sm.OLS(np.log(ysm), Xsm, missing='drop').fit()
    #fitd = fit.fittedvalues
    fitd = fit.predict(Xsm)
    if verbose:
        print(fit.summary())

    Xy = Xlag.join(ydf_test, how = 'inner')
    #good_til = np.logical_not(np.any(np.isnan(Xy), axis = 1))
    #Xy = Xy.loc[good_til,:]
    #Xs = Xy.drop(pred_col, axis = 1)
    Xs = Xy.loc[:,list(Xu.columns)+dow]
    #Xs = (Xs - mu_x[np.newaxis,:]) / sig_x[np.newaxis,:]#we did this already

    Xs = Xs.loc[:,~isconst]
    Xsm_test = sm.add_constant(Xs, has_constant = 'add')
    ysm_test = Xy[pred_col]

    if test_end is not None:
        Xsm_test = Xsm_test[Xsm_test.index<=test_end]
        ysm_test = ysm_test[ysm_test.index<=test_end]

    pred = fit.predict(Xsm_test)

    if lik == 'gaus':
        pred = np.exp(pred)
        fitd = np.exp(fitd)
    nmae_oos = -np.nanmean(np.abs(pred - ysm_test))
    nmae_is = -np.nanmean(np.abs(fitd - ysm))

    #return Xsm, ysm, fit, fitd, fit.llf, Xsm_test, ysm_test, pred, nmae
    return Xsm, ysm, fit, fitd, fit.rsquared, Xsm_test, ysm_test, pred, nmae_oos, nmae_is

#
#' max_lags int >= 0
#' max_window int >= 1
#' start_test pd.datetime after Feb 24 , and preferably after March 15ish
#' ydf - response dataframe + day of weeks
#' Xu - predictor dataframe
#' pred_col - which col of response data to use.
#' hump_quad - Should we fit a quadratic term wrt time during the initial period?
#' lik - one of 'gaus' (log-OLS), 'poisson' or 'nb' (for GLMs).
#' fname - Name of file showing lag/window cost surface.
#' pos_lag - Only consider positive lags? (i.e. if False allow retrodiction).
def est_lag(max_lags, max_window, start_test, ydf, Xu, pred_col = 'hps_outflow', hump_quad = False, lik = 'gaus', fname = 'cost.pdf', pos_lag = True, plot = False, test_end = None, verbose = False):
    lags = np.arange(max_lags) if pos_lag else np.arange(-max_lags,max_lags)
    nlags = len(lags)

    #TODO: Should we be using in-sample MSE instead(i.e. not on log scale)?
    r2s = np.zeros([nlags,max_window])
    ooss = np.zeros([nlags,max_window])
    lb = 0 if pos_lag else -max_lags
    for wi, w in enumerate(range(1,max_window+1)):
        for li, lag in enumerate(lags):
            if pos_lag and w > li:
                r2s[li,wi] = -np.inf
                ooss[li,wi] = -np.inf
            else:
                _, _, _, _, r2, _, _, _, nmae_oos, nmae_is = lag_n_reg(lag, w, start_test, ydf, Xu, pred_col, hump_quad, lik, test_end)
                r2s[li,wi] = r2
                ooss[li,wi] = nmae_oos

    if plot:
        fig = plt.figure()
        plt.subplot(2,1,1)
        plt.imshow(r2s, aspect = max_window/len(lags))
        plt.subplot(2,1,2)
        plt.imshow(ooss, aspect = max_window/len(lags))
        plt.savefig(fname)
        plt.close()

    # Select lag/window based on R2.
    llf_lagi, llf_wi = np.unravel_index(np.argmax(r2s), r2s.shape)
    opt_lag = lags[llf_lagi]
    opt_w = llf_wi+1

    Xsm, ysm, fit, fitd, r2, Xsm_test, ysm_test, pred, nmae_oos, nmae_is = lag_n_reg(opt_lag, opt_w, start_test, ydf, Xu, pred_col, hump_quad, lik, test_end, verbose = verbose)
    return opt_lag, opt_w, fit, nmae_oos, nmae_is
