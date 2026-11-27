#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer-expanded, leakage-controlled multichannel forecasting benchmark.

Added: SARIMA, TBATS, Ridge autoregression, HistGradientBoosting autoregression,
MLP neural autoregression, and ARFIMA(1,d,1) with jointly estimated (phi, theta, d, mu) via conditional
Gaussian likelihood (CSS). This is JOINT CONDITIONAL estimation, not exact MLE.

Install:
  python -m pip install numpy scipy pandas statsmodels scikit-learn tbats

TBATS is optional; if it is absent, its forecasts are marked unavailable.

Forecasts are computed at *the same* rolling origins across all horizons and
models; rolling_step only subsamples evaluation origins, never reuses stale
forecasts. Each forecast at time t uses observations <= t only.

Seasonality must correspond to sampling frequency: hourly s=24 (daily),
daily s=7 (weekly), monthly s=12. If the data have no convincing periodicity,
set SEASONAL_PERIODS=[] to disable the seasonal comparisons rather than force
spurious seasonal terms.
"""
from __future__ import annotations
import warnings
import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    from tbats import TBATS
    HAS_TBATS = True
except ImportError:
    TBATS = None
    HAS_TBATS = False

# ============================ USER SETTINGS ============================
# ============================
# USER SETTINGS
# ============================

MODE = "npy"

# ORIGINAL wind-speed measurements
NPY_PATH = "sp_2016_2d_arfima_vol.npy"

# Six spatially separated locations
CHANNEL_INDICES = [0, 1, 2]

# Your data have shape (1485, 8784)
ARRAY_LAYOUT = "CT"

MAX_TIME_POINTS = 10000

NORMALIZE_PER_CHANNEL = True

# Do not difference the original signals
PREPROCESS_LOGRET = False
PREPROCESS_ABS_LOGVOL = False

TRAIN_FRAC = 0.95

HORIZONS = (20, 30, 40, 50)

ROLLING_STEP = 10
MAX_ORIGINS = 40

FIT_WINDOW = 500

OUTPUT_DIR = "./results_presure_vol"

# Disable seasonal models for the initial comparison
SEASONAL_PERIODS = []
SARIMA_SEASONAL_PERIOD = None

ARMA_P_MAX = 1
ARMA_Q_MAX = 1

M_FRAC = 500
KERNEL_M = 400

ARFIMA_JOINT_ORDER = (1, 1)
ARFIMA_OPT_MAXITER = 100

ENABLE_TBATS = False
ENABLE_SARIMA = True
ENABLE_ML = True
SARIMA_MAXITER = 2

ML_LAGS = 48
ML_MAX_TRAIN_SAMPLES = 3000

RANDOM_SEED = 42
VERBOSE = True
D_TRUE = 0.4
SYNTHETIC_N = 2000
SYNTHETIC_BURN = 500
VALIDATION_FRAC = 0.0
EVAL_SPLIT = "test"
SELECTED_METHODS = None
# ======================================================================


def load_multichannel_npy(path, indices=None, layout="auto", max_points=None):
    x = np.load(path)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D/2D numpy data, got {x.shape}")
    if layout.upper() == "TC":
        x = x.T
    elif layout.upper() == "CT":
        pass
    elif layout.lower() == "auto":
        if x.shape[0] > x.shape[1]:
            x = x.T
    else:
        raise ValueError("layout must be 'auto', 'CT', or 'TC'")
    if indices is not None:
        x = x[indices, :]
    if max_points is not None:
        x = x[:, :max_points]
    return np.asarray(x, dtype=float)


def preprocess(x):
    if PREPROCESS_LOGRET and PREPROCESS_ABS_LOGVOL:
        raise ValueError("Select only one of log return and log volatility")
    if PREPROCESS_LOGRET or PREPROCESS_ABS_LOGVOL:
        if np.any(x <= 0):
            raise ValueError("Log preprocessing needs strictly positive RAW data")
        x = np.diff(np.log(x), axis=1)
        if PREPROCESS_ABS_LOGVOL:
            x = np.abs(x)
    return x


def fracdiff_weights(d, m):
    w = np.empty(m + 1, dtype=float)
    w[0] = 1.
    for j in range(1, m + 1):
        w[j] = w[j-1] * (j-1-d)/j
    return w


def frac_diff(x, d, m):
    return lfilter(fracdiff_weights(d, min(m, len(x)-1)), [1.], x)


def frac_reconstruct(yhat, x_hist, d, m):
    """Invert (1-L)^d x_t=y_t with actual historical x, no double truncation."""
    w = fracdiff_weights(d, min(m, len(x_hist) + len(yhat)))
    hist = list(np.asarray(x_hist, float))
    out = []
    for y in yhat:
        k = min(len(hist), len(w)-1)
        next_x = float(y - np.dot(w[1:k+1], np.asarray(hist[-k:][::-1]))) if k else float(y)
        hist.append(next_x)
        out.append(next_x)
    return np.asarray(out)


def hurst_dfa(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 64:
        return 0.5
    profile = np.cumsum(x-x.mean())
    scales = np.unique(np.geomspace(8, max(16, len(x)//4), 12).astype(int))
    Fs, Ss = [], []
    for s in scales:
        n = len(x)//s
        if n < 2:
            continue
        arr = profile[:n*s].reshape(n, s)
        tt = np.arange(s, dtype=float)
        tc = tt - tt.mean()
        denom = np.dot(tc, tc)
        slopes = ((arr - arr.mean(axis=1, keepdims=True)) @ tc)/denom
        trend = arr.mean(axis=1, keepdims=True) + slopes[:, None]*tc[None, :]
        f = np.sqrt(np.mean((arr-trend)**2))
        if f > 0 and np.isfinite(f):
            Fs.append(f); Ss.append(s)
    return float(np.clip(np.polyfit(np.log(Ss), np.log(Fs), 1)[0], .01, .99)) if len(Fs)>=4 else .5


def fit_arima(x, order):
    return ARIMA(x, order=order, trend="c", enforce_stationarity=True,
                 enforce_invertibility=True).fit(method_kwargs={"maxiter": 65})


def fc_naive(x,h):
    return np.full(h, x[-1], dtype=float)


def fc_mean(x,h):
    return np.full(h, x.mean(), dtype=float)


def fc_drift(x,h):
    return x[-1] + (x[-1]-x[0])/(len(x)-1)*np.arange(1,h+1)


def fc_arma(x,h):
    best, aic = None, np.inf
    for p in range(ARMA_P_MAX+1):
        for q in range(ARMA_Q_MAX+1):
            try:
                fit = fit_arima(x, (p,0,q))
                if fit.aic < aic and np.isfinite(fit.aic):
                    best, aic = fit, fit.aic
            except (ValueError, np.linalg.LinAlgError):
                continue
    if best is None:
        raise RuntimeError("All ARMA fits failed")
    return np.asarray(best.forecast(h))


def fc_ets(x,h):
    fit = ExponentialSmoothing(x, trend=None, seasonal=None,
                               initialization_method="estimated").fit(optimized=True)
    return np.asarray(fit.forecast(h))


def fc_kernel(x,h):
    eta = hurst_dfa(x)+.5
    m = min(KERNEL_M, len(x))
    k = np.arange(1, m+1)
    w = k.astype(float)**(-eta)
    w /= w.sum()
    hist = list(x)
    out=[]
    for _ in range(h):
        z = float(w @ np.asarray(hist[-m:][::-1]))
        hist.append(z); out.append(z)
    return np.asarray(out)


def fc_arfima_pipeline(x,h,d):
    # Use same mean convention in both fixed-d and DFA-d variants.
    mu = float(np.mean(x))
    xc = x-mu
    y = frac_diff(xc, d, M_FRAC)
    fit = fit_arima(y, (1,0,1))
    yhat = np.asarray(fit.forecast(h))
    return frac_reconstruct(yhat, xc, d, M_FRAC)+mu


def fc_arfima_fixed(x,h):
    return fc_arfima_pipeline(x,h,D_TRUE)


def fc_arfima_dfa(x,h):
    d = float(np.clip(hurst_dfa(x)-.5, -.49,.49))
    return fc_arfima_pipeline(x,h,d)


def arma_innovations(y, ar, ma):
    # phi(B) y = theta(B) epsilon, with theta(B)=1 + theta_1 B + ...
    return lfilter(np.r_[1.,-np.asarray(ar)], np.r_[1.,np.asarray(ma)], y)


def fc_arfima_joint_css(x,h):
    """Joint conditional Gaussian MLE (CSS) for ARFIMA(p,d,q).

    Optimizes mu, d, AR, MA *together* on the original training series.
    Truncates fractional differencing at M_FRAC and initializes unobserved
    presample values to zero. This is approximate CSS, NOT exact Gaussian MLE.
    """
    p,q = ARFIMA_JOINT_ORDER
    if len(x) < max(80,4*(p+q)+20):
        raise ValueError("Insufficient history for joint ARFIMA")
    sd = max(float(np.std(x)), 1e-8)
    centre = float(np.mean(x))
    z = (x-centre)/sd
    warmup = max(20, p+q+3)
    m = min(M_FRAC, len(x)-1)

    def objective(par):
        mu=par[0]; d=par[1]; ar=par[2:2+p]; ma=par[2+p:2+p+q]
        yd = frac_diff(z-mu, d, m)
        e = arma_innovations(yd, ar, ma)
        tail = e[warmup:]
        if not np.isfinite(tail).all():
            return 1e15
        variance = max(float(np.mean(tail*tail)), 1e-12)
        return len(tail)*np.log(variance)  # concentrated Gaussian CSS

    d0 = float(np.clip(hurst_dfa(z)-.5,-.4,.4))
    init = np.r_[0., d0, np.zeros(p+q)]
    bounds = [(-3.,3.),(-.49,.49)] + [(-.95,.95)]*(p+q)
    # Only (1,1) is supported with coefficient-wise stationarity/invertibility.
    if (p,q)!=(1,1):
        raise ValueError("Joint CSS currently supports order (p,q)=(1,1) only")
    candidates = [init, np.r_[0.,.0,np.zeros(p+q)]]
    fits=[]
    for start in candidates:
        res = minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": ARFIMA_OPT_MAXITER, "ftol":1e-7})
        if np.isfinite(res.fun):
            fits.append(res)
    if not fits:
        raise RuntimeError("Joint ARFIMA optimization failed")
    opt = min(fits, key=lambda r:r.fun)
    mu,d=opt.x[:2]; ar=opt.x[2:2+p]; ma=opt.x[2+p:2+p+q]
    y=frac_diff(z-mu,d,m)
    e=arma_innovations(y,ar,ma)
    yh=list(y)
    eh=list(e)
    pred_y=[]
    for _ in range(h):
        v=sum(ar[j]*yh[-j-1] for j in range(p)) + sum(ma[j]*eh[-j-1] for j in range(q))
        yh.append(float(v)); eh.append(0.)
        pred_y.append(float(v))
    predz=frac_reconstruct(np.asarray(pred_y),z-mu,d,m)+mu
    return predz*sd+centre


def fc_sarima(x,h):
    s = SARIMA_SEASONAL_PERIOD
    if s is None or s < 2:
        raise ValueError("SARIMA needs a meaningful seasonal period >=2")
    if len(x) < max(60, 3*s):
        raise ValueError("Not enough observations for seasonal ARIMA")
    candidates = [((1,0,1),(1,0,0,s)),
                  ((1,0,0),(1,0,1,s)),
                  ((1,0,1),(0,1,1,s))]
    best=None; best_aic=np.inf
    for order, seasonal_order in candidates:
        try:
            fitted = SARIMAX(x,order=order,seasonal_order=seasonal_order,
                             trend="c" if seasonal_order[1]==0 else "n",
                             enforce_stationarity=True,
                             enforce_invertibility=True).fit(disp=False,maxiter=SARIMA_MAXITER)
            if np.isfinite(fitted.aic) and fitted.aic<best_aic:
                best_aic=fitted.aic;best=fitted
        except (ValueError, np.linalg.LinAlgError):
            continue
    if best is None:
        raise RuntimeError("All SARIMA candidate fits failed")
    return np.asarray(best.forecast(steps=h))


def fc_tbats(x,h):
    if not HAS_TBATS:
        raise ImportError("TBATS unavailable: pip install tbats")
    # Disable Box-Cox (standardized series can be nonpositive).
    estimator=TBATS(seasonal_periods=SEASONAL_PERIODS,use_box_cox=False,
                    use_arma_errors=False,n_jobs=1)
    fit=estimator.fit(np.asarray(x,float))
    return np.asarray(fit.forecast(steps=h))


def lagged_matrix(x,lags):
    if len(x)<=lags+5:
        raise ValueError("Not enough values for requested ML_LAGS")
    from numpy.lib.stride_tricks import sliding_window_view
    mat=sliding_window_view(x,lags+1)
    return mat[:,:lags],mat[:,-1]


def fc_ml(x,h,method):
    # standardize on the *current training window* only
    mu=float(x.mean()); sd=max(float(x.std()),1e-8)
    z=(x-mu)/sd
    X,y=lagged_matrix(z,ML_LAGS)
    if len(X)>ML_MAX_TRAIN_SAMPLES:
        X,y=X[-ML_MAX_TRAIN_SAMPLES:],y[-ML_MAX_TRAIN_SAMPLES:]
    if method=="ridge":
        reg=Ridge(alpha=10.,fit_intercept=True)
    elif method=="mlp":
        reg=MLPRegressor(hidden_layer_sizes=(64,32),alpha=1.,max_iter=200,
                         early_stopping=True,n_iter_no_change=15,
                         random_state=RANDOM_SEED)
    elif method=="hgb":
        reg=HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,
                                          max_depth=4,learning_rate=.06,
                                          l2_regularization=1.,random_state=RANDOM_SEED)
    else:
        raise ValueError(method)
    reg.fit(X,y)
    hist=list(z)
    pred=[]
    for _ in range(h):
        v=float(reg.predict(np.asarray(hist[-ML_LAGS:])[None,:])[0])
        hist.append(v);pred.append(v)
    return np.asarray(pred)*sd+mu


def simulate_arfima(N=2000,d=.35,seed=42,burn=500):
    rng=np.random.default_rng(seed)
    eps=rng.standard_normal(N+burn)
    w=np.empty(N+burn);w[0]=1.
    for i in range(1,len(w)):
        w[i]=w[i-1]*(d+i-1)/i
    return lfilter(w,[1.],eps)[burn:]


@dataclass
class Score:
    channel: int
    name: str
    horizon: int
    rmse: float
    mae: float
    mase: float
    n_origins: int
    failures: int


def get_models(mode):
    models: Dict[str,Callable] = {
        "Naive":fc_naive,
        "Mean":fc_mean,
        "Drift":fc_drift,
        "ARMA(AIC grid)":fc_arma,
        "ETS(level)":fc_ets,
        "Kernel(DFA-H)":fc_kernel,
        "ARFIMA fixed-d":fc_arfima_fixed,
        "ARFIMA estimated-d(DFA)":fc_arfima_dfa,
        "ARFIMA joint-CSS(1,d,1)":fc_arfima_joint_css,
    }
    if mode=="synthetic":
        models[f"Kernel(oracle H={D_TRUE+.5:.2f})"] = lambda x,h: fc_kernel_oracle(x,h,D_TRUE+.5)
    if ENABLE_SARIMA and SARIMA_SEASONAL_PERIOD is not None:
        models[f"SARIMA(AIC,s={SARIMA_SEASONAL_PERIOD})"]=fc_sarima
    if ENABLE_TBATS and SEASONAL_PERIODS:
        models[f"TBATS(s={','.join(map(str,SEASONAL_PERIODS))})"]=fc_tbats
    if ENABLE_ML:
        models[f"Ridge-AR(lags={ML_LAGS})"] = lambda x,h: fc_ml(x,h,"ridge")
        models[f"HGB-AR(lags={ML_LAGS})"] = lambda x,h: fc_ml(x,h,"hgb")
        models[f"MLP-AR(lags={ML_LAGS})"] = lambda x,h: fc_ml(x,h,"mlp")
    return models


def fc_kernel_oracle(x,h,H):
    m=min(KERNEL_M,len(x));k=np.arange(1,m+1)
    w=k.astype(float)**(-H-.5);w/=w.sum()
    hist=list(x);o=[]
    for _ in range(h):
        a=float(w @ np.asarray(hist[-m:][::-1]))
        hist.append(a);o.append(a)
    return np.asarray(o)


def walk_forward_channel(x,channel,models):
    N=len(x);T0=int(TRAIN_FRAC*N);hmax=max(HORIZONS)
    T1=int((TRAIN_FRAC+VALIDATION_FRAC)*N)
    # Validation: origins from end of train through end of validation.
    # Test: origins from end of train+validation (or train if val_frac=0).
    if EVAL_SPLIT == "validation":
        start, stop = T0-1, T1
    else:
        start, stop = max(T0,T1)-1, N
    # origin t means observed x[:t+1]; targets x[t+1:t+1+h]
    # Restrict every target lead to the chosen evaluation partition.
    origins=list(range(start,stop-hmax+1,ROLLING_STEP))
    if MAX_ORIGINS is not None and len(origins)>MAX_ORIGINS:
        origins=np.linspace(0,len(origins)-1,MAX_ORIGINS).round().astype(int)
        origins=sorted(set((list(range(start,stop-hmax+1,ROLLING_STEP)))[i] for i in origins))
    if not origins:
        raise ValueError("No valid evaluation origins; inspect training split and horizons")
    t_true={h:[] for h in HORIZONS}
    preds=defaultdict(lambda:defaultdict(list)); failures=defaultdict(int)
    origin_records=[]; hurst_records=[]
    # normalization based ONLY on initial training period (no test leakage)
    initial=x[:T0]
    mu=float(np.mean(initial)); sd=max(float(np.std(initial)),1e-10)
    xx=(x-mu)/sd if NORMALIZE_PER_CHANNEL else x.copy()
    denominator=np.mean(np.abs(np.diff(xx[:T0])))
    for oi,t in enumerate(origins):
        train=xx[max(0,t+1-FIT_WINDOW):t+1] if FIT_WINDOW else xx[:t+1]
        truth=xx[t+1:t+1+hmax]
        for h in HORIZONS:
            t_true[h].append(truth[:h].copy())
        if "ARFIMA estimated-d(DFA)" in models:
            H=hurst_dfa(train)
            hurst_records.append({"channel":channel,"origin":t,"H_est_DFA":H,
                                  "d_est_DFA":float(np.clip(H-.5,-.49,.49)),
                                  "d_clipped":bool(H-.5<=-.49 or H-.5>=.49)})
        for name,model in models.items():
            tic=time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fc=np.asarray(model(train,hmax),dtype=float).ravel()
                if fc.size!=hmax or not np.all(np.isfinite(fc)):
                    raise ValueError(f"Invalid forecast length/values ({fc.size})")
            except Exception as exc:
                failures[name]+=1
                fc=np.full(hmax,np.nan)
                if VERBOSE and (failures[name]<=2):
                    print(f"    {name}: origin {oi} failed: {type(exc).__name__}: {exc}")
            elapsed=time.perf_counter()-tic
            for h in HORIZONS:
                valid=bool(np.isfinite(fc[:h]).all())
                err=truth[:h]-fc[:h] if valid else np.full(h,np.nan)
                origin_records.append({"channel":channel,"origin":t,"name":name,"horizon":h,
                                       "valid":valid,"runtime_seconds":elapsed,
                                       "mse":float(np.mean(err**2)),
                                       "mae":float(np.mean(np.abs(err)))})
                preds[name][h].append(fc[:h].copy())
        if VERBOSE and (oi+1)%10==0:
            print(f"   channel={channel} origins {oi+1}/{len(origins)}")
    rows=[]
    for name in models:
        for h in HORIZONS:
            truth=np.stack(t_true[h])
            estimated=np.stack(preds[name][h])
            good=np.isfinite(estimated).all(axis=1)
            # No silent naive imputation on failed model fit.
            if np.any(good):
                error=truth[good]-estimated[good]
                err_rmse=float(np.sqrt(np.mean(error**2)))
                err_mae=float(np.mean(np.abs(error)))
                err_mase=float(err_mae/denominator) if denominator>0 else np.nan
            else:
                err_rmse=err_mae=err_mase=np.nan
            rows.append(Score(channel,name,h,err_rmse,err_mae,err_mase,
                              int(good.sum()),failures[name]))
    return rows,origins,origin_records,hurst_records


def main():
    out=Path(OUTPUT_DIR);out.mkdir(parents=True,exist_ok=True)
    if MODE=="synthetic":
        data=simulate_arfima(N=SYNTHETIC_N,d=D_TRUE,seed=RANDOM_SEED,burn=SYNTHETIC_BURN)[None,:]
    elif MODE=="npy":
        data=load_multichannel_npy(NPY_PATH,CHANNEL_INDICES,ARRAY_LAYOUT,MAX_TIME_POINTS)
        data=preprocess(data)
    else:
        raise ValueError("MODE must be 'npy' or 'synthetic'")
    print(f"Data shape (channels,time) = {data.shape}; TBATS installed: {HAS_TBATS}")
    print(f"TRAIN_FRAC={TRAIN_FRAC}; horizons={HORIZONS}; origins-step={ROLLING_STEP}")
    print("Method: rolling-origin evaluation with no stale predictions")
    models=get_models(MODE)
    if SELECTED_METHODS is not None:
        unknown = [m for m in SELECTED_METHODS if m not in models]
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}. Available: {list(models)}")
        models = {m: models[m] for m in SELECTED_METHODS}
    rows=[]; all_origin_records=[]; all_hurst_records=[]
    for c,x in enumerate(data):
        if not np.isfinite(x).all() or np.std(x)<1e-12:
            print(f"Skipping invalid/constant channel {c}")
            continue
        print(f"--- Channel {c+1}/{len(data)} ---",flush=True)
        scores,origins,origin_records,hurst_records=walk_forward_channel(x,c,models)
        rows.extend(vars(r) for r in scores)
        all_origin_records.extend(origin_records)
        all_hurst_records.extend(hurst_records)
        pd.DataFrame({"origin_index":origins}).to_csv(out/f"origins_channel_{c}.csv",index=False)
    if not rows:
        raise RuntimeError("No valid channels")
    pd.DataFrame(all_origin_records).to_csv(out/"per_origin_scores.csv",index=False)
    pd.DataFrame(all_hurst_records,columns=["channel","origin","H_est_DFA","d_est_DFA","d_clipped"]).to_csv(out/"hurst_by_origin.csv",index=False)
    df=pd.DataFrame(rows)
    df.to_csv(out/"per_channel_scores.csv",index=False)
    agg=(df.groupby(["name","horizon"],as_index=False)
         .agg(rmse=("rmse","mean"),rmse_sd=("rmse","std"),
              mae=("mae","mean"),mae_sd=("mae","std"),
              mase=("mase","mean"),mase_sd=("mase","std"),
              n_channels=("rmse","count"),
              total_valid_origins=("n_origins","sum"),
              total_failures=("failures","sum")))
    agg.to_csv(out/"aggregated_scores.csv",index=False)
    for h in HORIZONS:
        table=agg.loc[agg.horizon==h].sort_values("rmse",na_position="last")
        print(f"\n=== Horizon h={h} | mean across channels ===")
        print(table[["name","rmse","rmse_sd","mae","mase","n_channels",
                     "total_valid_origins","total_failures"]].to_string(index=False,
                            float_format=lambda v:f"{v:.5f}"))
    print(f"\nSaved tables in {out.resolve()}")
    config = {key: _jsonable(globals()[key]) for key in CONFIG_KEYS}
    config["available_methods"] = list(models)
    (out/"run_config.json").write_text(json.dumps(config, indent=2))
    if ENABLE_TBATS and not HAS_TBATS:
        print("Note: TBATS was not installed. Install 'tbats' and rerun for TBATS results.")
    print("IMPORTANT: ARFIMA joint-CSS is approximate conditional likelihood, not exact ML.")

CONFIG_KEYS = ("MODE", "NPY_PATH", "CHANNEL_INDICES", "ARRAY_LAYOUT", "MAX_TIME_POINTS",
               "NORMALIZE_PER_CHANNEL", "PREPROCESS_LOGRET", "PREPROCESS_ABS_LOGVOL",
               "TRAIN_FRAC", "VALIDATION_FRAC", "EVAL_SPLIT", "HORIZONS", "ROLLING_STEP",
               "MAX_ORIGINS", "FIT_WINDOW", "OUTPUT_DIR", "SEASONAL_PERIODS",
               "SARIMA_SEASONAL_PERIOD", "ARMA_P_MAX", "ARMA_Q_MAX", "M_FRAC", "KERNEL_M",
               "ARFIMA_JOINT_ORDER", "ARFIMA_OPT_MAXITER", "ENABLE_TBATS", "ENABLE_SARIMA",
               "ENABLE_ML", "SARIMA_MAXITER", "ML_LAGS", "ML_MAX_TRAIN_SAMPLES",
               "RANDOM_SEED", "VERBOSE", "D_TRUE", "SYNTHETIC_N", "SYNTHETIC_BURN",
               "SELECTED_METHODS")


def _jsonable(v):
    if isinstance(v, tuple): return list(v)
    if isinstance(v, Path): return str(v)
    return v


def parse_args():
    p=argparse.ArgumentParser(description="Configurable ARFIMA multi-method rolling-origin benchmark")
    p.add_argument("--mode", choices=("npy","synthetic"), default=MODE)
    p.add_argument("--npy-path",default=NPY_PATH)
    p.add_argument("--channels",default=",".join(map(str,CHANNEL_INDICES)),
                   help="Comma-separated zero-based indices, or all")
    p.add_argument("--array-layout",choices=("CT","TC","auto"),default=ARRAY_LAYOUT)
    p.add_argument("--max-time-points",type=int,default=MAX_TIME_POINTS)
    p.add_argument("--train-frac",type=float,default=TRAIN_FRAC)
    p.add_argument("--val-frac",type=float,default=VALIDATION_FRAC)
    p.add_argument("--eval-split",choices=("validation","test"),default=EVAL_SPLIT)
    p.add_argument("--horizons",default=",".join(map(str,HORIZONS)))
    p.add_argument("--rolling-step",type=int,default=ROLLING_STEP)
    p.add_argument("--max-origins",type=int,default=MAX_ORIGINS,
                   help="0 means all origins")
    p.add_argument("--fit-window",type=int,default=FIT_WINDOW,
                   help="0 means expanding history")
    p.add_argument("--output-dir",default=OUTPUT_DIR)
    p.add_argument("--arma-p-max",type=int,default=ARMA_P_MAX)
    p.add_argument("--arma-q-max",type=int,default=ARMA_Q_MAX)
    p.add_argument("--m-frac",type=int,default=M_FRAC)
    p.add_argument("--kernel-m",type=int,default=KERNEL_M)
    p.add_argument("--joint-p",type=int,default=ARFIMA_JOINT_ORDER[0])
    p.add_argument("--joint-q",type=int,default=ARFIMA_JOINT_ORDER[1])
    p.add_argument("--joint-maxiter",type=int,default=ARFIMA_OPT_MAXITER)
    p.add_argument("--sarima-maxiter",type=int,default=SARIMA_MAXITER)
    p.add_argument("--seasonal-periods",default=",".join(map(str,SEASONAL_PERIODS)))
    p.add_argument("--sarima-period",type=int,default=SARIMA_SEASONAL_PERIOD)
    p.add_argument("--enable-sarima",action=argparse.BooleanOptionalAction,default=ENABLE_SARIMA)
    p.add_argument("--enable-tbats",action=argparse.BooleanOptionalAction,default=ENABLE_TBATS)
    p.add_argument("--enable-ml",action=argparse.BooleanOptionalAction,default=ENABLE_ML)
    p.add_argument("--normalize",action=argparse.BooleanOptionalAction,default=NORMALIZE_PER_CHANNEL)
    p.add_argument("--preprocess-logret",action="store_true",default=PREPROCESS_LOGRET)
    p.add_argument("--preprocess-abs-logvol",action="store_true",default=PREPROCESS_ABS_LOGVOL)
    p.add_argument("--ml-lags",type=int,default=ML_LAGS)
    p.add_argument("--ml-max-train-samples",type=int,default=ML_MAX_TRAIN_SAMPLES)
    p.add_argument("--seed",type=int,default=RANDOM_SEED)
    p.add_argument("--d-true",type=float,default=D_TRUE)
    p.add_argument("--synthetic-n",type=int,default=SYNTHETIC_N)
    p.add_argument("--synthetic-burn",type=int,default=SYNTHETIC_BURN)
    p.add_argument("--methods",default="all",help="Comma-separated exact model names; all for full benchmark")
    p.add_argument("--verbose",action=argparse.BooleanOptionalAction,default=VERBOSE)
    a=p.parse_args()
    if not (0<a.train_frac<1 and 0<=a.val_frac<1 and a.train_frac+a.val_frac<1):
        p.error("Need 0 < train-frac < 1 and 0 <= val-frac with train-frac+val-frac < 1")
    if a.eval_split=="validation" and a.val_frac==0:
        p.error("--eval-split validation requires --val-frac > 0")
    if a.rolling_step<1 or a.max_origins<0 or a.fit_window<0 or a.m_frac<1 or a.kernel_m<1:
        p.error("rolling-step >= 1, max-origins >= 0, fit-window >= 0, filter lengths >= 1 required")
    if a.arma_p_max<0 or a.arma_q_max<0 or a.synthetic_n<100 or a.synthetic_burn<0:
        p.error("Invalid ARMA order or synthetic series length/burn")
    if not (-.49<a.d_true<.49): p.error("--d-true must be in (-0.49,0.49)")
    return a


def apply_args(a):
    updates={
      "MODE":a.mode,"NPY_PATH":a.npy_path,
      "CHANNEL_INDICES":None if a.channels.lower()=="all" else [int(z) for z in a.channels.split(",")],
      "ARRAY_LAYOUT":a.array_layout,"MAX_TIME_POINTS":a.max_time_points or None,
      "NORMALIZE_PER_CHANNEL":a.normalize,
      "PREPROCESS_LOGRET":a.preprocess_logret,"PREPROCESS_ABS_LOGVOL":a.preprocess_abs_logvol,
      "TRAIN_FRAC":a.train_frac,"VALIDATION_FRAC":a.val_frac,"EVAL_SPLIT":a.eval_split,
      "HORIZONS":tuple(int(z) for z in a.horizons.split(",")),
      "ROLLING_STEP":a.rolling_step,"MAX_ORIGINS":a.max_origins or None,
      "FIT_WINDOW":a.fit_window or None,"OUTPUT_DIR":a.output_dir,
      "ARMA_P_MAX":a.arma_p_max,"ARMA_Q_MAX":a.arma_q_max,
      "M_FRAC":a.m_frac,"KERNEL_M":a.kernel_m,
      "ARFIMA_JOINT_ORDER":(a.joint_p,a.joint_q),"ARFIMA_OPT_MAXITER":a.joint_maxiter,
      "SARIMA_MAXITER":a.sarima_maxiter,
      "SEASONAL_PERIODS":[] if not a.seasonal_periods else [int(z) for z in a.seasonal_periods.split(",")],
      "SARIMA_SEASONAL_PERIOD":a.sarima_period,
      "ENABLE_TBATS":a.enable_tbats,"ENABLE_SARIMA":a.enable_sarima,"ENABLE_ML":a.enable_ml,
      "ML_LAGS":a.ml_lags,"ML_MAX_TRAIN_SAMPLES":a.ml_max_train_samples,
      "RANDOM_SEED":a.seed,"VERBOSE":a.verbose,"D_TRUE":a.d_true,
      "SYNTHETIC_N":a.synthetic_n,"SYNTHETIC_BURN":a.synthetic_burn,
      "SELECTED_METHODS":None if a.methods=="all" else [z.strip() for z in a.methods.split(",")]
    }
    if not updates["HORIZONS"] or min(updates["HORIZONS"])<1:
        raise ValueError("HORIZONS must be positive integers")
    if a.joint_p!=1 or a.joint_q!=1:
        raise ValueError("Current joint-CSS implementation only supports --joint-p 1 --joint-q 1")
    globals().update(updates)


if __name__=="__main__":
    apply_args(parse_args())
    main()
