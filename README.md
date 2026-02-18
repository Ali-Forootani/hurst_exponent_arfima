Here is a research-paper-style `README.md` written in academic tone and structure.

---

# ARFIMA Multi-Channel Forecasting Benchmark

*A systematic evaluation of fractional long-memory models versus classical time-series baselines*

---

## Abstract

This repository provides a unified experimental framework for evaluating **ARFIMA-based long-memory forecasting models** against classical time-series baselines.

The framework supports:

* Synthetic ARFIMA(0, d, 0) data generation
* Multi-channel real-world `.npy` datasets
* Hurst exponent estimation via DFA
* Fractional differencing and inversion
* Walk-forward multi-horizon forecasting
* Aggregated performance across channels

The objective is to assess whether fractional long-memory structure provides measurable forecasting gains relative to ARMA, ETS, and kernel-based approaches.

---

## 1. Motivation

Many financial, geophysical, and climatological time series exhibit **long-range dependence**, often characterized by:

[
\text{Cov}(X_t, X_{t+k}) \sim k^{2H-2}
]

where ( H \in (0,1) ) is the **Hurst exponent**.

ARFIMA models extend ARIMA by allowing fractional integration:

[
(1 - L)^d X_t = \varepsilon_t
]

where:

[
d = H - 0.5
]

This project benchmarks:

* Fixed fractional memory models
* Data-driven memory estimation (via DFA)
* Classical short-memory alternatives

under controlled and real-data settings.

---

## 2. Methodology

### 2.1 Fractional Operators

Fractional differencing:

[
(1 - L)^d = \sum_{k=0}^{\infty} w_k L^k
]

Weights computed recursively:

[
w_k = w_{k-1} \frac{k-1-d}{k}
]

Fractional integration:

[
(1 - L)^{-d}
]

Used to invert forecasts step-by-step.

---

### 2.2 ARFIMA Forecasting Pipeline

1. Apply fractional differencing
2. Fit ARMA(p,q) selected by AIC grid search
3. Forecast in the transformed domain
4. Invert via fractional integration

Two variants are implemented:

* **ARFIMA fixed-d**
* **ARFIMA estimated-d (DFA-based)**

---

### 2.3 Hurst Estimation (DFA)

Detrended Fluctuation Analysis (DFA(1)):

[
F(s) \sim s^H
]

Slope of log-log regression provides ( H ).

Memory parameter:

[
d = \text{clip}(H - 0.5, -0.49, 0.49)
]

---

### 2.4 Kernel-Based Long-Memory Baseline

Power-law kernel:

[
w_k \propto k^{-\eta}
]

with

[
\eta = \frac{3 - 2H}{2}
]

Provides a non-parametric long-memory benchmark.

---

## 3. Models Compared

### Long-Memory

* ARFIMA fixed-d
* ARFIMA estimated-d (DFA)
* Kernel (DFA-H)
* Kernel (oracle H) — synthetic mode only

### Short-Memory / Classical

* Naive
* Mean
* Drift
* ARMA (AIC grid search)
* ETS (level-only exponential smoothing)

---

## 4. Experimental Design

### Walk-Forward Validation

* Expanding training window
* Multi-horizon forecasts
* Configurable refit frequency

Default horizons:

```
(1, 5, 10, 20)
```

Training fraction:

```
train_frac = 0.8
```

---

## 5. Multi-Channel Support

The framework accepts `.npy` files with shapes:

* `(C, T)`
* `(T, C)` (auto-detected)

Channel selection supports:

```python
CHANNEL_INDICES = [0, 1]
CHANNEL_INDICES = slice(0, 10)
CHANNEL_INDICES = None
```

Scores are computed per channel and aggregated via mean.

Channels containing:

* NaN
* Infinite values
* Constant signals

are excluded.

---

## 6. Preprocessing Options

Optional transformations:

### Absolute Log-Volatility

[
| \Delta \log X_t |
]

### Log-Returns

[
\Delta \log X_t
]

### Per-Channel Standardization

[
\frac{X - \mu}{\sigma}
]

---

## 7. Performance Metrics

For each horizon ( h ):

* RMSE
* MAE
* MASE

MASE defined as:

[
\text{MASE} = \frac{\text{MAE}}{\text{MAE of naive one-step}}
]

Models are ranked by RMSE per horizon.

---

## 8. Synthetic Benchmark

Simulates ARFIMA(0, d, 0):

```python
d_true = 0.35
```

Includes oracle H kernel for calibration.

Used to validate:

* Correct fractional inversion
* Recovery of memory structure
* Comparative forecasting gain

---

## 9. Real-Data Benchmark

Set:

```python
MODE = "npy"
NPY_PATH = "crypto_coins.npy"
```

The oracle kernel is automatically disabled.

---

## 10. Installation

```bash
pip install numpy statsmodels
```

Python ≥ 3.8 recommended.

Run:

```bash
python ARFIMA_multichannel_benchmark_indices.py
```

---

## 11. Expected Use Cases

* Financial long-memory analysis
* Volatility modeling
* Climate or wind-speed persistence studies
* Fractional vs short-memory comparative research
* Hurst-based model validation

---

## 12. Limitations

* ARFIMA estimation approximated via fractional filtering + ARMA
* No full maximum-likelihood fractional estimation
* DFA may be biased in small samples
* Computational cost increases with large fractional truncation ( M )

---

## 13. Reproducibility

Synthetic mode is seeded:

```python
seed=42
```

For deterministic experiments.

---

## 14. Research Context

This benchmark enables empirical investigation of:

* When does long memory improve forecasting?
* Does DFA-based memory estimation help?
* How does fractional modeling compare to ARMA in practice?
* Is kernel-based memory competitive with ARFIMA?

---

## 15. Author

Ali Forootani, email:aliforootani@ieee.org, forootani@gea.mpg.de
2026

---

