#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 11:47:39 2026

@author: forootani
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARFIMA readiness test script (single file)

What this does (per channel):
1) Loads a .npy file and forces shape (C,T)
2) Cleans NaN/Inf
3) Runs DFA-H on:
   - raw series x
   - first difference dx = diff(x)
   - (optional) log-returns if all x>0
   - (optional) volatility proxy |diff(log(x))|
4) Runs quick stationarity diagnostics:
   - ADF test (unit root)  [p small => stationary]
   - KPSS test (trend-stationary null) [p small => nonstationary]
5) Prints a decision suggestion:
   - "ARIMA(d=1) likely" if raw H≈1 and H(diff)≈0.5 and ADF fails on raw
   - "ARFIMA candidate" if raw looks stationary-ish and H>0.55 and H(diff) stays >0.55
   - "Short-memory / ARMA" if H(diff) near 0.5
   - "Anti-persistent" if H<0.5

Run:
  python arfima_readiness_test.py

Edit:
  NPY_PATH, CHANNEL_SLICE, and optional flags below.
"""

import numpy as np
import warnings
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller, kpss
from pathlib import Path

# =========================
# CONFIG
# =========================
NPY_PATH = "crypto_coins_2d_arfima_vol.npy"

# choose channels: None (all) or slice/list
CHANNEL_SLICE = slice(0, 5)  # e.g. slice(0,6) or [0,1,2] or None

# DFA settings
S_MIN = 16
N_SCALES = 28
FIT_TOP_FRAC = 0.5
MIN_BLOCKS = 12

# Diagnostics
DO_LOG_RETURNS_TESTS = True   # only runs if all values > 0 in channel
PLOT_CHANNEL = 0              # None to disable plotting
PLOT_MAX_T = 5000             # truncate for plotting only
# =========================


# ----------------------------
# Helper: ensure array is (C, T)
# ----------------------------
def as_channels_first(data: np.ndarray, transpose_if_needed: bool = True) -> np.ndarray:
    """
    Return data as shape (C, T).
    Accepts:
      - (T,) -> (1, T)
      - (T, C) or (C, T) -> heuristic transpose so output is (C, T)
    """
    data = np.asarray(data)
    if data.ndim == 1:
        return data.reshape(1, -1).astype(float)

    if data.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {data.shape}")

    X = data.astype(float)
    
    
    
    print(data.shape)
    

    # heuristic: time dimension is usually larger
    if transpose_if_needed and X.shape[0] > X.shape[1]:
        # likely (T, C) -> transpose to (C, T)
        X = X.T
        
    #    print("after transpose")
    #    print(X.shape)
        
    return X


# ----------------------------
# DFA Hurst with min-blocks fix
# ----------------------------
def hurst_dfa_minblocks(
    x: np.ndarray,
    *,
    order: int = 1,
    s_min: int = 16,
    n_scales: int = 28,
    fit_top_frac: float = 0.5,
    min_blocks: int = 12,
    s_cap: int | None = None,
    plot: bool = False,
    title: str = "",
):
    x = np.asarray(x, float).ravel()
    n = len(x)
    if n < 4 * s_min:
        return 0.5, np.array([]), np.array([]), (None, None)

    y = np.cumsum(x - x.mean())

    s_max = max(32, n // 4)
    scales = np.unique(np.logspace(np.log10(s_min), np.log10(s_max), n_scales).astype(int))
    scales = scales[(scales >= s_min) & (scales < n)]

    # enforce enough blocks per scale
    ok_scales = []
    for s in scales:
        if (n // s) >= min_blocks:
            ok_scales.append(s)
    scales = np.asarray(ok_scales, int)

    if s_cap is not None:
        scales = scales[scales <= int(s_cap)]

    if len(scales) < 8:
        return 0.5, scales, np.array([]), (None, None)

    # compute F(s)
    F = []
    for s in scales:
        m = n // s
        yy = y[:m * s].reshape(m, s)
        t = np.arange(s)

        rms2 = []
        for b in yy:
            c = np.polyfit(t, b, order)
            trend = np.polyval(c, t)
            rms2.append(np.mean((b - trend) ** 2))
        f = np.sqrt(np.mean(rms2))
        F.append(f if np.isfinite(f) and f > 0 else np.nan)

    F = np.asarray(F, float)
    ok = np.isfinite(F) & (F > 0)
    scales, F = scales[ok], F[ok]

    if len(F) < 8:
        return 0.5, scales, F, (None, None)

    # fit top fraction (largest scales)
    k0 = int(np.floor((1 - fit_top_frac) * len(scales)))
    k0 = np.clip(k0, 0, len(scales) - 2)

    s_fit = scales[k0:]
    F_fit = F[k0:]

    H = np.polyfit(np.log(s_fit), np.log(F_fit), 1)[0]
    H = float(np.clip(H, 0.01, 0.99))

    if plot:
        plt.figure()
        plt.loglog(scales, F, marker="o", linestyle="-", label="F(s)")
        coef = np.polyfit(np.log(s_fit), np.log(F_fit), 1)
        plt.loglog(s_fit, np.exp(np.polyval(coef, np.log(s_fit))), "--",
                   label=f"fit top {int(fit_top_frac*100)}% | H={H:.3f}")
        plt.xlabel("scale s")
        plt.ylabel("F(s)")
        plt.title(title if title else f"DFA scaling (H≈{H:.3f})")
        plt.grid(True, which="both", ls="--", alpha=0.4)
        plt.legend()
        plt.show()

    return H, scales, F, (int(s_fit[0]), int(s_fit[-1]))


# ----------------------------
# Stationarity tests (ADF/KPSS)
# ----------------------------
def adf_pvalue(x: np.ndarray) -> float:
    # H0: unit root (nonstationary). small p => reject => stationary
    try:
        return float(adfuller(x, autolag="AIC")[1])
    except Exception:
        return np.nan

def kpss_pvalue(x: np.ndarray) -> float:
    # H0: (trend-)stationary. small p => reject => nonstationary
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(kpss(x, regression="ct", nlags="auto")[1])
    except Exception:
        return np.nan


# ----------------------------
# Decision heuristic
# ----------------------------
def suggest_model(H_raw, H_diff, p_adf_raw, p_adf_diff, p_kpss_raw, p_kpss_diff):
    """
    Returns a short text suggestion. This is a heuristic, not a proof.
    """
    # classic random-walk symptom
    if (H_raw > 0.85) and (0.45 <= H_diff <= 0.60):
        return "Looks integrated/trending. Difference once (ARIMA d=1) or detrend; then ARMA/ARFIMA on residual."

    # stationary + persistent
    if (H_raw > 0.55) and (H_diff > 0.55):
        return "Persistent across differencing. ARFIMA candidate (fractional d > 0)."

    # short memory
    if 0.45 <= H_diff <= 0.60:
        return "Increments look short-memory. ARMA on differenced/returns may be enough."

    # anti-persistent
    if H_raw < 0.45 or H_diff < 0.45:
        return "Anti-persistent/mean-reverting behavior. ARMA or ARFIMA with negative d may fit."

    return "Mixed signals; inspect DFA plot across scale ranges and consider de-seasonalizing/detrending."


# ----------------------------
# Main analysis
# ----------------------------
def analyze_channel(x: np.ndarray, ch_name: str = "", plot: bool = False):
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    if len(x) < 1024 or np.std(x) < 1e-12:
        return {
            "channel": ch_name,
            "n": len(x),
            "H_raw": np.nan,
            "H_diff": np.nan,
            "adf_raw": np.nan,
            "adf_diff": np.nan,
            "kpss_raw": np.nan,
            "kpss_diff": np.nan,
            "suggestion": "Too short/constant."
        }

    H_raw, *_ = hurst_dfa_minblocks(
        x, s_min=S_MIN, n_scales=N_SCALES, fit_top_frac=FIT_TOP_FRAC, min_blocks=MIN_BLOCKS,
        plot=plot, title=f"{NPY_PATH} | {ch_name} | raw"
    )
    dx = np.diff(x)
    H_diff, *_ = hurst_dfa_minblocks(
        dx, s_min=S_MIN, n_scales=N_SCALES, fit_top_frac=FIT_TOP_FRAC, min_blocks=MIN_BLOCKS,
        plot=plot, title=f"{NPY_PATH} | {ch_name} | diff"
    )

    p_adf_raw = adf_pvalue(x)
    p_adf_diff = adf_pvalue(dx)
    p_kpss_raw = kpss_pvalue(x)
    p_kpss_diff = kpss_pvalue(dx)

    suggestion = suggest_model(H_raw, H_diff, p_adf_raw, p_adf_diff, p_kpss_raw, p_kpss_diff)

    out = {
        "channel": ch_name,
        "n": int(len(x)),
        "H_raw": float(H_raw),
        "H_diff": float(H_diff),
        "adf_raw": float(p_adf_raw),
        "adf_diff": float(p_adf_diff),
        "kpss_raw": float(p_kpss_raw),
        "kpss_diff": float(p_kpss_diff),
        "suggestion": suggestion
    }

    # Optional: log-returns and volatility if x>0
    if DO_LOG_RETURNS_TESTS and np.all(x > 0):
        r = np.diff(np.log(x))
        v = np.abs(r)
        Hr, *_ = hurst_dfa_minblocks(r, s_min=S_MIN, n_scales=N_SCALES, fit_top_frac=FIT_TOP_FRAC, min_blocks=MIN_BLOCKS)
        Hv, *_ = hurst_dfa_minblocks(v, s_min=S_MIN, n_scales=N_SCALES, fit_top_frac=FIT_TOP_FRAC, min_blocks=MIN_BLOCKS)
        out["H_logret"] = float(Hr)
        out["H_abslogret"] = float(Hv)
    else:
        out["H_logret"] = np.nan
        out["H_abslogret"] = np.nan

    return out


def main():
    data = np.load(NPY_PATH,  allow_pickle=True)
    
    
    filename = Path(NPY_PATH).name.lower()
    
    transpose_if_needed = True
    
    # Transpose if filename indicates wind/pressure OR fallback to your heuristic
    if transpose_if_needed:
        if ("wind_speed_2020" in filename) or ("pressure_surface_2020" in filename):
            data = data.T
            
            X = data
            
        else:
            
            X = as_channels_first(data, transpose_if_needed=True)  # (C,T)
    
    
    
    #X = as_channels_first(data, transpose_if_needed=True)  # (C,T)
    print(f"\n=== {NPY_PATH} ===")
    print("Raw shape:", data.shape, "-> using (C,T):", X.shape)

    if CHANNEL_SLICE is not None:
        X = X[CHANNEL_SLICE, :18000]
        print("Selected channels -> shape:", X.shape)

    # quick plot of selected channels (levels)
    if PLOT_CHANNEL is not None and 0 <= int(PLOT_CHANNEL) < X.shape[0]:
        tmax = min(PLOT_MAX_T, X.shape[1])
        plt.figure()
        plt.plot(X[int(PLOT_CHANNEL), :tmax])
        plt.title(f"{NPY_PATH} | channel {int(PLOT_CHANNEL)} | level (first {tmax})")
        plt.grid(True, alpha=0.3)
        plt.show()

    results = []
    for c in range(X.shape[0]):
        plot_this = (PLOT_CHANNEL is not None and c == int(PLOT_CHANNEL))
        res = analyze_channel(X[c], ch_name=f"ch{c}", plot=plot_this)
        results.append(res)

    # Print nicely
    print("\nResults (per channel):")
    header = (
        f"{'ch':>4} {'n':>6} | {'H_raw':>6} {'H_diff':>6} | "
        f"{'ADF(raw)':>8} {'ADF(diff)':>9} | {'KPSS(raw)':>9} {'KPSS(diff)':>10} | "
        f"{'H_logret':>7} {'H_|logret|':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['channel']:>4} {r['n']:6d} | "
            f"{r['H_raw']:6.3f} {r['H_diff']:6.3f} | "
            f"{r['adf_raw']:8.3g} {r['adf_diff']:9.3g} | "
            f"{r['kpss_raw']:9.3g} {r['kpss_diff']:10.3g} | "
            f"{r['H_logret']:7.3f} {r['H_abslogret']:10.3f}"
        )
        print("      ->", r["suggestion"])

    # Aggregate quick summary
    H_raws = np.array([r["H_raw"] for r in results], float)
    H_diffs = np.array([r["H_diff"] for r in results], float)
    print("\nSummary:")
    print("Mean H_raw (finite): ", float(np.nanmean(H_raws)))
    print("Mean H_diff(finite): ", float(np.nanmean(H_diffs)))
    print("\nHeuristic takeaway:")
    if np.nanmean(H_raws) > 0.85 and (0.45 <= np.nanmean(H_diffs) <= 0.60):
        print("- On average this looks integrated/trending: difference once before ARFIMA/ARMA.")
    elif np.nanmean(H_raws) > 0.55 and np.nanmean(H_diffs) > 0.55:
        print("- On average persistence survives differencing: ARFIMA is plausible.")
    elif 0.45 <= np.nanmean(H_diffs) <= 0.60:
        print("- On average increments are short-memory: ARMA may suffice.")
    elif np.nanmean(H_raws) < 0.45 or np.nanmean(H_diffs) < 0.45:
        print("- On average anti-persistent: consider ARFIMA with negative d or ARMA.")
    else:
        print("- Mixed: inspect per-channel suggestions and DFA plots.")


if __name__ == "__main__":
    main()
