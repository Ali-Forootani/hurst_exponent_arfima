#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 11:06:35 2026

@author: forootani
"""

"""
ARFIMA benchmark suite (single file) with MULTI-CHANNEL .npy loader + channel selection by indices

Key update you asked for:
- load_multichannel_npy(..., indices=[...]) to select channels like your PyTorch code pattern.

Supports:
- .npy shaped (channels, T) OR (T, channels)  -> returns (channels, T)
- indices can be list[int] or slice or None (all channels)

Models:
- Naive / Mean / Drift
- ARMA(AIC grid)
- ETS(level)
- Power-law kernel using DFA-H (and oracle H only in synthetic mode)
- ARFIMA pipeline fixed d
- ARFIMA pipeline estimated d via DFA

Run:
  python ARFIMA_multichannel_benchmark_indices.py

Typical usage for real data:
  MODE="npy"
  NPY_PATH="crypto_coins.npy"
  CHANNEL_INDICES=[0,1,2,5,7]  # select channels
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pathlib import Path


# ============================================================
# FRACTIONAL OPERATORS
# ============================================================
def fracdiff_weights(d: float, M: int) -> np.ndarray:
    """
    Weights for (1-L)^d (fractional differencing), truncated at lag M.
    Recursion produces: w_k = (-1)^k * binom(d, k)
    """
    w = np.zeros(M + 1, dtype=float)
    w[0] = 1.0
    for k in range(1, M + 1):
        w[k] = w[k - 1] * (k - 1 - d) / k
    return w

def fracintegrate_weights(d: float, M: int) -> np.ndarray:
    """
    Weights for (1-L)^(-d) (fractional integration / inverse filter), truncated at lag M.
    psi_k = binom(d+k-1, k)
    """
    psi = np.zeros(M + 1, dtype=float)
    psi[0] = 1.0
    for k in range(1, M + 1):
        psi[k] = psi[k - 1] * (d + k - 1) / k
    return psi

def frac_difference(x: np.ndarray, d: float, M: int) -> np.ndarray:
    """
    y_t = sum_{k=0}^M w_k x_{t-k}, with w from (1-L)^d
    """
    w = fracdiff_weights(d, M)
    y = np.empty_like(x, dtype=float)
    for t in range(len(x)):
        kmax = min(t, M)
        # x[t], x[t-1], ..., x[t-kmax]
        y[t] = np.dot(w[: kmax + 1], x[t - np.arange(kmax + 1)])
    return y

def invert_one_step_from_yhat(yhat: float, y_hist: np.ndarray, d: float, M: int) -> float:
    """
    Invert one step using (1-L)^(-d):
      x_{t+1} = sum_{k=0}^M psi_k y_{t+1-k}
    where y_{t+1} is yhat and y_hist contains y_0..y_t.

    IMPORTANT FIX:
    - Use y_hist[-k] (most recent) not y_hist[0] (oldest).
    """
    psi = fracintegrate_weights(d, M)
    x_next = psi[0] * float(yhat)

    kmax = min(M, len(y_hist))
    for k in range(1, kmax + 1):
        # k=1 uses last observed y_t, k=2 uses y_{t-1}, ...
        x_next += psi[k] * float(y_hist[-k])

    return float(x_next)


# ============================================================
# SYNTHETIC DATA: ARFIMA(0,d,0)
# ============================================================
def simulate_arfima_0d0(N: int, d: float, burn: int = 400, M: int = 900, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(N + burn)
    psi = fracintegrate_weights(d, M)
    x = np.zeros(N + burn, dtype=float)
    for t in range(N + burn):
        kmax = min(t, M)
        x[t] = np.dot(psi[: kmax + 1], eps[t - np.arange(kmax + 1)])
    return x[burn:]


# ============================================================
# MULTI-CHANNEL .NPY LOADER + CHANNEL SELECTION
# ============================================================
ChannelIndex = Union[None, List[int], np.ndarray, slice, Tuple[int, ...]]

def load_multichannel_npy(
    path: str,
    *,
    indices: ChannelIndex = None,
    transpose_if_needed: bool = True,
) -> np.ndarray:
    """
    Load multi-channel time series from a .npy file, and optionally select channels.

    Accepts:
      - Stored shape (C, T) or (T, C) (heuristic transpose so output is (C, T))
      - indices:
          * None        -> all channels
          * list[int]   -> select those channels
          * np.ndarray  -> int array of channel indices
          * tuple[int]  -> select those channels
          * slice       -> select slice of channels

    Returns:
      data (C_sel, T)
    """
    data = np.load(path)
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {data.shape}")

    # Heuristic: time dimension usually larger, so if rows > cols treat as (T,C) and transpose.
    #if transpose_if_needed and data.shape[0] > data.shape[1]:
    #    data = data.T
        
        
    filename = Path(path).name.lower()

    # Transpose if filename indicates wind/pressure OR fallback to your heuristic
    if transpose_if_needed:
        if ("wind_speeds_2020" in filename) or ("pressure_surface_2020" in filename):
            data = data.T
        elif data.shape[0] > data.shape[1]:  # fallback heuristic
            data = data.T

    data = data.astype(float)  # (C, T) after heuristic

    # Channel selection (no forced truncation here; do it outside if needed)
    if indices is not None:
        data = data[indices, :10000]

    if data.ndim != 2:
        raise ValueError(f"After indexing, expected 2D (C,T); got shape {data.shape}")

    return data

def minmax_to_unit_interval(data, eps=1e-8):
    """
    Per-channel min-max normalization to (eps, 1-eps)
    Avoids exact 0 and 1 for log stability.
    """
    data = data.astype(float)

    mn = data.min(axis=1, keepdims=True)
    mx = data.max(axis=1, keepdims=True)

    denom = mx - mn
    denom[denom == 0] = 1.0

    z = (data - mn) / denom
    z = z * (1 - 2*eps) + eps

    return z


# ============================================================
# DFA HURST ESTIMATOR (simple DFA(1))
# ============================================================
def hurst_dfa(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    n = len(x)
    if n < 64:
        return 0.5

    y = np.cumsum(x - x.mean())
    scales = np.unique(np.logspace(np.log10(8), np.log10(max(16, n // 4)), 12).astype(int))

    Fs, Ss = [], []
    for s in scales:
        if s < 4 or s >= n:
            continue
        m = n // s
        if m < 2:
            continue

        blocks = y[: m * s].reshape(m, s)
        t = np.arange(s)

        rms = []
        for b in blocks:
            c = np.polyfit(t, b, 1)
            trend = c[0] * t + c[1]
            rms.append(np.mean((b - trend) ** 2))

        F = np.sqrt(np.mean(rms))
        if np.isfinite(F) and F > 0:
            Fs.append(F)
            Ss.append(s)

    if len(Fs) < 4:
        return 0.5

    H = np.polyfit(np.log(Ss), np.log(Fs), 1)[0]
    return float(np.clip(H, 0.01, 0.99))

def d_from_H(H: float) -> float:
    return float(np.clip(H - 0.5, -0.49, 0.49))


# ============================================================
# ROBUST ARMA FITTING (AIC grid search)
# ============================================================
def try_fit_arma(y: np.ndarray, order: Tuple[int, int], method: str = "statespace", maxiter: int = 200):
    p, q = order
    try:
        fit = ARIMA(
            y,
            order=(p, 0, q),
            trend="n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        ).fit(method=method, maxiter=maxiter, disp=0)

        if hasattr(fit, "mle_retvals") and isinstance(fit.mle_retvals, dict):
            if not fit.mle_retvals.get("converged", True):
                return None
        return fit
    except Exception:
        return None

def fit_best_arma_robust(y: np.ndarray, p_max: int = 3, q_max: int = 3, method: str = "statespace"):
    best_aic = np.inf
    best_fit = None

    for p in range(p_max + 1):
        for q in range(q_max + 1):
            if p == 0 and q == 0:
                continue
            fit = try_fit_arma(y, (p, q), method=method)
            if fit is None:
                continue
            if fit.aic < best_aic:
                best_aic = fit.aic
                best_fit = fit

    if best_fit is None:
        best_fit = ARIMA(y, order=(0, 0, 0), trend="n").fit()

    return best_fit


# ============================================================
# METRICS
# ============================================================
def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat)))

def mase(y: np.ndarray, yhat: np.ndarray, insample: np.ndarray) -> float:
    insample = np.asarray(insample, float)
    denom = np.mean(np.abs(np.diff(insample)))
    if denom == 0:
        return float("inf")
    return float(mae(y, yhat) / denom)


# ============================================================
# FORECASTERS (univariate)
# ============================================================
def fc_naive(x: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(float(x[-1]), h).astype(float)

def fc_mean(x: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(float(np.mean(x)), h).astype(float)

def fc_drift(x: np.ndarray, h: int) -> np.ndarray:
    if len(x) < 2:
        return fc_naive(x, h)
    drift = (x[-1] - x[0]) / (len(x) - 1)
    return (x[-1] + drift * np.arange(1, h + 1)).astype(float)

def fc_arma(x: np.ndarray, h: int, p_max: int = 3, q_max: int = 3) -> np.ndarray:
    fit = fit_best_arma_robust(x, p_max=p_max, q_max=q_max)
    return np.asarray(fit.forecast(h), float)

def fc_ets_level(x: np.ndarray, h: int) -> np.ndarray:
    fit = ExponentialSmoothing(
        x, trend=None, seasonal=None, initialization_method="estimated"
    ).fit(optimized=True)
    return np.asarray(fit.forecast(h), float)

def fc_powerlaw_kernel(x: np.ndarray, h: int, H: float, M: int = 400) -> np.ndarray:
    """
    Power-law kernel baseline:
      w_k ∝ k^{-eta}, eta = (3-2H)/2 = (1+alpha)/2
    """
    eta = (3.0 - 2.0 * float(H)) / 2.0
    M_eff = min(M, len(x))
    k = np.arange(1, M_eff + 1)
    w = k ** (-eta)
    w /= w.sum()

    preds = []
    hist = x.astype(float).copy()
    for _ in range(h):
        nxt = float(np.dot(w, hist[-k]))
        preds.append(nxt)
        hist = np.append(hist, nxt)
    return np.asarray(preds, float)

def fc_arfima_fixed_d(
    x: np.ndarray,
    h: int,
    d: float = 0.25,
    M_frac: int = 400,
    p_max: int = 3,
    q_max: int = 3,
) -> np.ndarray:
    """
    ARFIMA pipeline approximation:
      1) y = (1-L)^d x  (fractional differencing)
      2) fit ARMA on y
      3) forecast y for h steps
      4) invert step-by-step using (1-L)^(-d) to obtain x forecasts
    """
    y = frac_difference(x, d, M_frac)
    fit = fit_best_arma_robust(y, p_max=p_max, q_max=q_max)

    # IMPORTANT FIX: forecast the full h-step path in y-domain once
    yhat_seq = np.asarray(fit.forecast(h), float)

    preds = []
    y_hist = y.astype(float).copy()
    for i in range(h):
        yhat = float(yhat_seq[i])
        xhat = invert_one_step_from_yhat(yhat, y_hist, d=d, M=M_frac)
        preds.append(xhat)
        y_hist = np.append(y_hist, yhat)

    return np.asarray(preds, float)

def fc_arfima_estimated_d(
    x: np.ndarray,
    h: int,
    M_frac: int = 400,
    p_max: int = 3,
    q_max: int = 3,
) -> np.ndarray:
    H_hat = hurst_dfa(x)
    d_hat = d_from_H(H_hat)
    return fc_arfima_fixed_d(x, h, d=d_hat, M_frac=M_frac, p_max=p_max, q_max=q_max)


# ============================================================
# WALK-FORWARD (univariate)
# ============================================================
@dataclass
class Score:
    name: str
    h: int
    rmse: float
    mae: float
    mase: float

def walk_forward_univariate(
    x: np.ndarray,
    models: Dict[str, Callable[[np.ndarray, int], np.ndarray]],
    horizons: Tuple[int, ...] = (1, 5, 10, 20),
    train_frac: float = 0.7,
    refit_every: int = 1,
) -> List[Score]:
    x = np.asarray(x, float)
    N = len(x)
    T0 = int(train_frac * N)

    scores: List[Score] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)

        for h in horizons:
            yt = {name: [] for name in models}
            yp = {name: [] for name in models}

            cache_pred = {name: None for name in models}
            cache_t = -10**9

            for t in range(T0, N - h):
                train = x[: t + 1]
                true = x[t + 1 : t + 1 + h]

                do_refit = (t - cache_t) >= refit_every or any(cache_pred[n] is None for n in models)
                if do_refit:
                    for name, fc in models.items():
                        try:
                            pred = np.asarray(fc(train, h), float).reshape(-1)
                            if pred.shape[0] != h:
                                pred = pred[:h] if pred.shape[0] > h else np.pad(pred, (0, h - pred.shape[0]), mode="edge")
                            cache_pred[name] = pred
                        except Exception:
                            cache_pred[name] = fc_naive(train, h)
                    cache_t = t

                for name in models:
                    yt[name].append(true)
                    yp[name].append(cache_pred[name])

            for name in models:
                y_true = np.vstack(yt[name])
                y_pred = np.vstack(yp[name])
                scores.append(
                    Score(
                        name=name,
                        h=h,
                        rmse=rmse(y_true, y_pred),
                        mae=mae(y_true, y_pred),
                        mase=mase(y_true, y_pred, x[:T0]),
                    )
                )

    return scores


# ============================================================
# MULTI-CHANNEL BENCHMARK + AGGREGATION
# ============================================================
def aggregate_scores(score_lists: List[List[Score]]) -> List[Score]:
    bucket: Dict[Tuple[str, int], List[Score]] = {}
    for scores in score_lists:
        for s in scores:
            bucket.setdefault((s.name, s.h), []).append(s)

    out: List[Score] = []
    for (name, h), vals in bucket.items():
        out.append(
            Score(
                name=name,
                h=h,
                rmse=float(np.mean([v.rmse for v in vals])),
                mae=float(np.mean([v.mae for v in vals])),
                mase=float(np.mean([v.mase for v in vals])),
            )
        )
    return out

def benchmark_multichannel(
    data: np.ndarray,
    models: Dict[str, Callable[[np.ndarray, int], np.ndarray]],
    horizons: Tuple[int, ...] = (1, 5, 10, 20),
    train_frac: float = 0.7,
    refit_every: int = 10,
    verbose: bool = True,
) -> List[Score]:
    data = np.asarray(data, float)
    if data.ndim != 2:
        raise ValueError("data must be 2D (channels, T)")

    C, _T = data.shape
    per_channel_scores: List[List[Score]] = []

    for ch in range(C):
        x = data[ch]
        if not np.isfinite(x).all():
            if verbose:
                print(f"Channel {ch}: NaN/inf -> skip")
            continue
        if np.std(x) < 1e-10:
            if verbose:
                print(f"Channel {ch}: constant -> skip")
            continue

        if verbose:
            print(f"--- Channel {ch+1}/{C} ---")

        s = walk_forward_univariate(
            x, models, horizons=horizons, train_frac=train_frac, refit_every=refit_every
        )
        per_channel_scores.append(s)

    if len(per_channel_scores) == 0:
        raise RuntimeError("No valid channels were benchmarked.")

    return aggregate_scores(per_channel_scores)


# ============================================================
# PRINTING
# ============================================================
def print_scores(scores: List[Score], title: str = "") -> None:
    if title:
        print(title)
    horizons = sorted(set(s.h for s in scores))
    for h in horizons:
        rows = sorted([s for s in scores if s.h == h], key=lambda z: z.rmse)
        print(f"\n=== Horizon h={h} ===")
        print(f"{'Model':32s} {'RMSE':>10s} {'MAE':>10s} {'MASE':>10s}")
        for r in rows:
            print(f"{r.name:32s} {r.rmse:10.5f} {r.mae:10.5f} {r.mase:10.5f}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # ----------------------------
    # Choose mode
    # ----------------------------
    MODE = "npy"  # "synthetic" or "npy"

    # For MODE="npy"
    NPY_PATH = "crypto_coins.npy"
    
    # rlc_both_realistic.npy
    
    # Select channels like your indices pattern:
    # Examples:
    #   CHANNEL_INDICES = [0, 1, 2, 5, 7]
    #   CHANNEL_INDICES = slice(0, 10)
    #   CHANNEL_INDICES = None  # all
    CHANNEL_INDICES: ChannelIndex = [0, 1]

    # Data preprocessing switches (OFF by default; enable as appropriate)
    PREPROCESS_ABS_LOGVOL = False   # abs(diff(log(x))) volatility proxy (requires x>0)
    PREPROCESS_LOGRET = False      # diff(log(x)) returns (requires x>0)
    NORMALIZE_PER_CHANNEL = True

    # Benchmark settings
    horizons = (1, 5, 10, 20)
    train_frac = 0.8

    # Speed: 1 is slowest; 5-10 is much faster.
    refit_every = 5

    # ARMA / ARFIMA internal settings
    p_max, q_max = 2, 2
    M_frac = 1000

    # Synthetic-only parameters (oracle kernel)
    d_true = 0.35
    H_oracle = d_true + 0.5

    # ----------------------------
    # Define models
    # ----------------------------
    models: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
        "Naive": fc_naive,
        "Mean": fc_mean,
        "Drift": fc_drift,
        "ARMA(AIC grid)": lambda x, h: fc_arma(x, h, p_max=p_max, q_max=q_max),
        "ETS(level)": fc_ets_level,
        "Kernel(DFA-H)": lambda x, h: fc_powerlaw_kernel(x, h, H=hurst_dfa(x), M=400),
        f"Kernel(oracle H={H_oracle:.2f})": (lambda x, h: fc_powerlaw_kernel(x, h, H=H_oracle, M=400)),
        "ARFIMA fixed-d": lambda x, h: fc_arfima_fixed_d(x, h, d= d_true, M_frac=M_frac, p_max=p_max, q_max=q_max),
        "ARFIMA estimated-d(DFA)": lambda x, h: fc_arfima_estimated_d(x, h, M_frac=M_frac, p_max=p_max, q_max=q_max),
    }

    if MODE == "npy":
        # Oracle kernel doesn't make sense on real data; remove.
        models.pop(f"Kernel(oracle H={H_oracle:.2f})", None)

    # ----------------------------
    # Run
    # ----------------------------
    if MODE == "synthetic":
        x = simulate_arfima_0d0(N=2000, d=d_true, burn=400, M=900, seed=42)
        scores = walk_forward_univariate(
            x, models, horizons=horizons, train_frac=train_frac, refit_every=1
        )
        print_scores(scores, title="\nSYNTHETIC BENCHMARK")

    elif MODE == "npy":
        data = load_multichannel_npy(NPY_PATH, indices=CHANNEL_INDICES)
        print("Loaded .npy shape (selected channels, T):", data.shape)

        # Preprocessing (choose ONE appropriate transform)
            
        if PREPROCESS_ABS_LOGVOL:
            # Ensure strictly positive BEFORE log
            data = minmax_to_unit_interval(data, eps=1e-8)   # now in (eps, 1-eps), strictly >0
            logx = np.log(data)
            data = np.abs(np.diff(logx, axis=1))             # volatility proxy; can be 0
            print("Converted to abs log-vol proxy. New shape:", data.shape)
    
        

        elif PREPROCESS_LOGRET:
            if np.any(data <= 0):
                raise ValueError("PREPROCESS_LOGRET requires strictly positive data (for log).")
            data = np.diff(np.log(data), axis=1)
            print("Converted to log-returns. New shape:", data.shape)

        if NORMALIZE_PER_CHANNEL:
            mu = data.mean(axis=1, keepdims=True)
            sd = data.std(axis=1, keepdims=True)
            sd[sd == 0] = 1.0
            data = (data - mu) / sd
            print("Normalized per channel.")

        scores = benchmark_multichannel(
            data,
            models,
            horizons=horizons,
            train_frac=train_frac,
            refit_every=refit_every,
            verbose=True,
        )
        print_scores(scores, title="\nMULTI-CHANNEL BENCHMARK (aggregated across selected channels)")

    else:
        raise ValueError("MODE must be 'synthetic' or 'npy'")
