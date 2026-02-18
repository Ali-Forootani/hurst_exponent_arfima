#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 18:27:28 2026

@author: forootani
"""

import numpy as np
import os
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
INPUT_NPY = "df_cleaned_numeric_2.npy"   # change to your dataset
OUTPUT_PREFIX = "df_cleaned_numeric_2"

CHANNELS = None          # None = all OR slice(0,7) OR [0,1,2]
T_MAX = None             # e.g. 20000 or None

STANDARDIZE = True
REMOVE_MEAN = True

EPS = 1e-12


# ============================================================
# Helpers
# ============================================================
def as_channels_first(X):
    X = np.asarray(X)

    if X.ndim == 1:
        return X.reshape(1, -1).astype(float)

    if X.shape[0] > X.shape[1]:
        return X.T.astype(float)

    return X.astype(float)


def fill_nan_1d(x):
    x = x.copy()
    n = len(x)

    mask = np.isfinite(x)
    if mask.sum() < 2:
        x[:] = np.nanmean(x) if np.isfinite(np.nanmean(x)) else 0
        return x

    idx = np.arange(n)
    x[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
    return x


def clean_matrix(X):
    X = X.copy()
    X[~np.isfinite(X)] = np.nan
    for c in range(X.shape[0]):
        X[c] = fill_nan_1d(X[c])
    return X


def standardize_per_channel(Y):
    if REMOVE_MEAN:
        Y = Y - Y.mean(axis=1, keepdims=True)

    if STANDARDIZE:
        sd = Y.std(axis=1, keepdims=True)
        sd[sd == 0] = 1
        Y = Y / sd

    return Y


# ============================================================
# MAIN PREPROCESS
# ============================================================
print("\nLoading:", INPUT_NPY)
X = np.load(INPUT_NPY)
X = as_channels_first(X)

if CHANNELS is not None:
    X = X[CHANNELS]

if T_MAX is not None:
    X = X[:, :T_MAX]

print("Input shape (C,T):", X.shape)

X = clean_matrix(X)

# ============================================================
# Track 1 — Mean dynamics
# ============================================================
Y_mean = np.diff(X, axis=1)
Y_mean = standardize_per_channel(Y_mean)

plt.plot(Y_mean[0,:200])

# ============================================================
# Track 2 — Long-memory volatility
# ============================================================
Y_vol = np.abs(np.diff(X, axis=1))
Y_vol = standardize_per_channel(Y_vol)

plt.plot(Y_vol[0,:1000])


# ============================================================
# SAVE OUTPUTS
# ============================================================
mean_path = OUTPUT_PREFIX + "_arfima_mean.npy"
vol_path  = OUTPUT_PREFIX + "_arfima_vol.npy"

np.save(mean_path, Y_mean.astype(np.float64))
np.save(vol_path,  Y_vol.astype(np.float64))

print("\nSaved:")
print("Mean track :", mean_path, Y_mean.shape)
print("Vol  track :", vol_path,  Y_vol.shape)

print("\nThese are ready for ARFIMA (C,T).")
