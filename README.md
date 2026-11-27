# ARFIMA Multi-Channel Forecasting Benchmark

*A systematic, rolling-origin evaluation of fractional long-memory models, classical statistical methods, and machine-learning autoregressors*

## Abstract

This repository provides a configurable experimental framework for assessing the predictive utility of fractional long-memory structure in synthetic and real-world multichannel time series. Three ARFIMA formulations are compared: fixed fractional integration, DFA-based memory estimation, and joint conditional Gaussian estimation of ARFIMA(1, d, 1). The benchmark incorporates classical statistical, seasonal, kernel-based, and machine-learning baselines.

The evaluation uses common forecasting origins and multiple prediction horizons, with either fixed-length rolling training windows or expanding histories. Preprocessing and normalization are designed to prevent future observations from entering model fitting. Model failures are recorded explicitly rather than replaced with naive predictions. The framework exports channel-level, origin-level, and cross-channel performance statistics, Hurst estimates, runtimes, and experiment configurations for reproducible comparative and sensitivity studies.

## 1. Motivation

Financial, geophysical, climatological, and other complex time series may exhibit persistent temporal dependence. For a stationary long-memory process with Hurst exponent $H\in(0.5,1)$, the autocovariance may decay asymptotically as

$$
\gamma(k)\sim C k^{2H-2},\qquad k\to\infty.
$$

The ARFIMA family generalizes integer-order integration through

$$
\Phi(L)(1-L)^d(X_t-\mu)=\Theta(L)\varepsilon_t,
$$

where $L$ is the lag operator, $\Phi$ and $\Theta$ are the autoregressive and moving-average polynomials, and $d$ is the fractional memory parameter. For stationary ARFIMA processes with appropriate ARMA factors, $H=d+1/2$ characterizes the low-frequency scaling behavior.

The benchmark examines whether long-memory modeling improves multi-step forecasting relative to short-memory, seasonal, and nonlinear data-driven alternatives, and whether estimating $d$ jointly with the ARMA parameters provides advantages over a two-stage DFA-based approach.

## 2. Methodology

### 2.1 Fractional differencing and reconstruction

The fractional differencing operator admits the expansion

$$
(1-L)^d=\sum_{k=0}^{\infty}w_k L^k,
\qquad w_0=1,\qquad
w_k=w_{k-1}\frac{k-1-d}{k}.
$$

The implementation uses a causal, truncated filter with maximum lag `M_FRAC` (default 500), further limited by the available history during filtering. Let $Y_t=(1-L)^d(X_t-\mu)$ and $Z_t=X_t-\mu$. Given forecasts in the transformed domain, reconstruction proceeds recursively:

$$
\widehat Z_{t+j}=\widehat Y_{t+j}
-\sum_{k=1}^{m}w_k\widetilde Z_{t+j-k},
\qquad
\widehat X_{t+j}=\widehat Z_{t+j}+\mu,
$$

where unavailable presample terms are omitted, and $\widetilde Z$ denotes observed centered values for times up to the forecast origin and recursively reconstructed predictions thereafter. This directly inverts the finite differencing recurrence without a second, independently truncated integration filter.

### 2.2 Two-stage ARFIMA forecasting

Both two-stage variants use the same estimation and forecasting pipeline:

1. Estimate the sample mean from the current fitting window and center the observations.
2. Apply the truncated fractional differencing operator.
3. Fit ARMA(1, 1) to the filtered signal, with a constant term and enforced stationarity/invertibility.
4. Generate forecasts in the transformed domain.
5. Reconstruct forecasts recursively and restore the original window mean.

The two variants differ only in their memory parameter:

- **ARFIMA fixed-d:** uses the configured `D_TRUE` (default 0.4). On real data this is a prescribed parameter, not a known ground truth.
- **ARFIMA estimated-d(DFA):** estimates $d$ from the observations in each fitting window.

Unlike the standalone ARMA baseline, these two-stage ARFIMA methods use a fixed ARMA(1, 1) order rather than AIC-based order selection.

### 2.3 Detrended fluctuation analysis

First-order detrended fluctuation analysis (DFA-1) estimates the scaling exponent from

$$
F(s)\propto s^H.
$$

The centered series is cumulatively summed, partitioned into nonoverlapping segments at approximately logarithmically spaced scales (from 8 to roughly one quarter of the window length), and linearly detrended within each segment. The root-mean-square residual fluctuation is computed at each valid scale. The slope of the log-log regression estimates $H$.

The estimate is restricted to $[0.01,0.99]$; when fewer than 64 observations or fewer than four valid scales are available, the implementation returns $H=0.5$. The fractional parameter is computed as

$$
\widehat d=\operatorname{clip}(\widehat H-0.5,-0.49,0.49).
$$

DFA is computed separately using the current training history at each origin. The estimated exponent is a proxy for long-memory behavior, not a statistical test establishing its presence.

### 2.4 Joint conditional ARFIMA(1, d, 1) estimation

The joint-CSS baseline estimates the mean, memory parameter, AR coefficient, and MA coefficient simultaneously. After standardizing the current fitting window to obtain $z_t$, the model is expressed as

$$
(1-\phi L)(1-L)^d(z_t-\mu)
=(1+\theta L)\varepsilon_t.
$$

For each candidate parameter vector $\boldsymbol\vartheta=(\mu,d,\phi,\theta)$, conditional innovations are computed by causal filtering. Presample states are initialized to zero. Discarding the initial warm-up innovations, the concentrated conditional Gaussian objective is

$$
J(\boldsymbol\vartheta)
=n_e\log\left(
\max\left\{
\frac{1}{n_e}\sum_{t\in\mathcal I}\varepsilon_t(\boldsymbol\vartheta)^2,
10^{-12}
\right\}
\right),
$$

where $\mathcal I$ is the retained innovation index set and $n_e=|\mathcal I|$. At the supported order (1, 1), the first 20 innovations are discarded.

The bounded L-BFGS-B optimizer uses two initializations: DFA-based $d$ (restricted to $[-0.4,0.4]$) and $d=0$, with the remaining initial parameters set to zero. The candidate with the lowest finite objective is retained. Parameter bounds are

| Parameter | Bounds |
|---|---|
| $\mu$ (standardized domain) | $[-3,3]$ |
| $d$ | $[-0.49,0.49]$ |
| $\phi,\theta$ | $[-0.95,0.95]$ |

At least 80 observations are required for the implemented order. The transformed-domain forecast is generated recursively with future innovations set to zero, followed by fractional reconstruction and inverse scaling.

**Important:** this is approximate joint conditional Gaussian likelihood / conditional sum-of-squares (CSS) estimation with finite fractional truncation, not exact Gaussian maximum likelihood. The current implementation supports only ARFIMA(1, d, 1), and a finite optimizer objective does not by itself establish convergence.

### 2.5 Power-law kernel forecasting

The kernel baseline constructs a normalized, nonnegative weighting scheme

$$
a_k=\frac{k^{-\eta}}{\sum_{j=1}^{m}j^{-\eta}},
\qquad \eta=H+\frac12,
$$

with $m$ bounded by `KERNEL_M` (default 400). The most recent observation receives the weight $a_1$. Predictions are generated recursively by a weighted average of the observed and previously predicted values.

The DFA-based kernel is available for all datasets; an oracle-H kernel using $H=D_{\mathrm{TRUE}}+0.5$ is available only in synthetic mode.

### 2.6 Classical and seasonal statistical models

The standalone ARMA baseline selects $(p,q)$ via minimum finite AIC over the grid $0\le p\le P_{\max}$, $0\le q\le Q_{\max}$, fitting ARIMA$(p,0,q)$ with a constant term. Naive, sample mean, linear drift, and level-only exponential smoothing are included as additional references.

When a meaningful seasonal period $s$ is specified, SARIMA fits the following candidate orders and selects the smallest finite AIC:

| Nonseasonal order | Seasonal order |
|---|---|
| (1,0,1) | (1,0,0,s) |
| (1,0,0) | (1,0,1,s) |
| (1,0,1) | (0,1,1,s) |

SARIMA requires at least $\max(60,3s)$ training observations. TBATS is an optional seasonal benchmark supporting multiple seasonal periods. The implemented configuration disables Box–Cox transformations and ARMA errors and uses a single worker.

Seasonality must be justified by the data sampling frequency and domain knowledge (e.g., hourly daily cycles $s=24$, daily weekly cycles $s=7$, or monthly annual cycles $s=12$). Seasonal models are not included in the current default run.

### 2.7 Machine-learning autoregression

Machine-learning baselines use a univariate lag vector of length $\ell$ and estimate the one-step mapping

$$
\widehat z_{t+1}=f_\theta(z_{t-\ell+1},\ldots,z_t).
$$

All ML training features and targets are standardized using the current fitting window only. At most `ML_MAX_TRAIN_SAMPLES` recent lagged training pairs are retained. A single fitted model generates multi-step forecasts recursively, and predictions are mapped back to the input scale.

| Model | Configuration |
|---|---|
| Ridge-AR | $\ell=48$, ridge penalty $\alpha=10$, intercept |
| HGB-AR | Histogram gradient boosting; 100 iterations, maximum 15 leaves, depth 4, learning rate 0.06, L2 regularization 1 |
| MLP-AR | Hidden layers (64, 32), L2 penalty 1, maximum 200 iterations, early stopping, patience 15 |

The MLP and HGB random states are controlled by `RANDOM_SEED`. MLP early stopping uses scikit-learn's internal validation mechanism on lagged samples from the fitting window, not a separately defined chronological validation partition.

## 3. Models compared

| Category | Method name in result files | Availability |
|---|---|---|
| Classical | `Naive` | Default |
| Classical | `Mean` | Default |
| Classical | `Drift` | Default |
| Classical | `ARMA(AIC grid)` | Default |
| Classical | `ETS(level)` | Default |
| Long memory | `Kernel(DFA-H)` | Default |
| Long memory | `ARFIMA fixed-d` | Default |
| Long memory | `ARFIMA estimated-d(DFA)` | Default |
| Long memory | `ARFIMA joint-CSS(1,d,1)` | Default |
| Long memory | `Kernel(oracle H=0.90)` | Synthetic only; label depends on d |
| Seasonal | `SARIMA(AIC,s=24)` | Conditional; example period |
| Seasonal | `TBATS(s=24)` | Optional dependency and specified periods |
| Machine learning | `Ridge-AR(lags=48)` | Default |
| Machine learning | `HGB-AR(lags=48)` | Default |
| Machine learning | `MLP-AR(lags=48)` | Default |

The default real-data configuration evaluates 12 methods. Synthetic mode adds the oracle-H kernel; seasonal methods can be enabled separately. Model names containing periods or lag counts change with their settings.

## 4. Experimental design

### 4.1 Chronological rolling-origin evaluation

For a series of length $N$, define $T_0=\lfloor f_{\mathrm{train}}N\rfloor$ and $T_1=\lfloor(f_{\mathrm{train}}+f_{\mathrm{val}})N\rfloor$. The initial training partition is $[0,T_0)$, optional validation is $[T_0,T_1)$, and test is $[T_1,N)$.

At each forecast origin $t$, the model receives only observations through $t$. With fitting window length $W$, the available training history is

$$
\mathcal X_t=(X_{\max(0,t-W+1)},\ldots,X_t).
$$

Setting `FIT_WINDOW=0` through the CLI selects an expanding history. Otherwise, the default uses a rolling window of 500 observations.

All models are refitted at each selected origin, generate one forecast trajectory of length $h_{\max}$, and use its prefixes for each requested horizon. The rolling step subsamples evaluation origins rather than reusing previously computed forecasts. If necessary, `MAX_ORIGINS` caps the number of origins using approximately evenly spaced subsampling across the eligible origin list.

The benchmark is designed to use origins for which all requested forecast targets lie in the selected chronological evaluation partition. See the boundary caveat in Section 11 before running arbitrary split and stride combinations.

### 4.2 Current default configuration

| Setting | Default |
|---|---|
| Mode | `npy` |
| Dataset | `sp_2016_2d_arfima_vol.npy` |
| Selected channels (zero-based) | 0, 1, 2 |
| Array layout | `CT` |
| Maximum time points | 10,000 |
| Initial training fraction | 0.95 |
| Validation fraction | 0.0 |
| Evaluation partition | Test |
| Forecast horizons | 20, 30, 40, 50 |
| Origin stride | 10 |
| Maximum origins per channel | 40 |
| Rolling fitting window | 500 |
| Fractional filter maximum lag | 500 |
| Kernel maximum lag | 400 |
| ARMA search maximum orders | (1, 1) |
| Fixed fractional parameter | 0.4 |
| Joint-CSS maximum optimizer iterations | 100 |
| ML lags / maximum training pairs | 48 / 3,000 |
| Normalization | Initial-training statistics |
| Log preprocessing | Disabled |
| Seasonal periods | None |
| SARIMA enabled flag / maximum iterations | True / 2 (inactive without period) |
| TBATS | Disabled |
| Machine-learning methods | Enabled |
| Random seed | 42 |
| Output directory | `./results_presure_vol` |

These are implementation defaults, not universal recommended hyperparameters. In particular, the SARIMA iteration default of 2 is useful only as a very limited fitting budget; increase it and assess convergence for substantive seasonal comparisons.

## 5. Multi-channel datasets and preprocessing

The framework accepts a one-dimensional NumPy series or a two-dimensional `.npy` array:

- `CT`: (channels, time).
- `TC`: (time, channels), transposed internally.
- `auto`: assumes the larger dimension is time; use an explicit layout for ambiguous arrays.

Each selected channel is modeled independently; this is a multichannel evaluation framework, not a joint multivariate predictor. Channel selection uses zero-based indices or `all` on the command line. The optional time cap retains the first specified number of observations. Nonfinite or effectively constant channels are skipped after preprocessing; log preprocessing requires strictly positive raw values.

Optional preprocessing for strictly positive signals includes either log returns,

$$
r_t=\log X_t-\log X_{t-1},
$$

or absolute log returns (a volatility proxy), $v_t=|r_t|$. These transforms are mutually exclusive and reduce the sequence length by one. Forecasts and evaluation remain in the selected transformed domain; the script does not convert log-return forecasts back to raw levels.

When enabled, each channel is standardized with the mean and standard deviation of its **initial training partition only**:

$$
X_t^{\mathrm{std}}=\frac{X_t-\mu_{\mathrm{train}}}{\sigma_{\mathrm{train}}}.
$$

The same transformation is applied to all evaluation observations. ML and joint-CSS models additionally scale their current fitting windows internally and invert that local scaling before scoring. Reported RMSE and MAE are in the standardized domain when normalization is enabled; MASE is scale invariant under nondegenerate affine scaling.

## 6. Performance metrics and statistical aggregation

For model $m$, channel $c$, horizon $h$, and set $\mathcal O_{m,c}$ of successful origins, let

$$
e_{c,t,j}=X_{c,t+j}-\widehat X_{c,t+j},
\qquad j=1,\ldots,h.
$$

The reported channel-level metrics are

$$
\mathrm{RMSE}_{m,c,h}=
\sqrt{\frac{1}{h|\mathcal O_{m,c}|}
\sum_{t\in\mathcal O_{m,c}}\sum_{j=1}^{h}e_{c,t,j}^{2}},
$$

$$
\mathrm{MAE}_{m,c,h}=
\frac{1}{h|\mathcal O_{m,c}|}
\sum_{t\in\mathcal O_{m,c}}\sum_{j=1}^{h}|e_{c,t,j}|,
$$

$$
\mathrm{MASE}_{m,c,h}=
\frac{\mathrm{MAE}_{m,c,h}}
{\frac{1}{T_0-1}\sum_{i=1}^{T_0-1}|X_{c,i}-X_{c,i-1}|}.
$$

MASE uses the one-step naive error calculated exclusively on the initial training partition. If its denominator is zero, MASE is undefined and recorded as NaN.

**Horizon convention:** each metric evaluates all leads 1 through $h$ jointly, not just the endpoint at lead $h$. This makes the horizon statistics cumulative and overlapping.

Channel-level scores are aggregated by method and horizon using the arithmetic mean and sample standard deviation across channels with available scores. This is a macro-average, not a pooled error over all observations. The printed rankings sort mean RMSE in ascending order.

Model fitting or forecast failures are not replaced by a naive forecast. An exception, incorrect forecast length, or nonfinite forecast produces a missing prediction and a recorded failure. Scores use successful origins only; methods can therefore have different effective samples even though their intended origins coincide. For strict paired comparisons, filter the origin-level records to the intersection of successful origins across the relevant methods. Inspect coverage and failures alongside accuracy.

## 7. Synthetic benchmark

Synthetic mode generates a Gaussian ARFIMA(0, d, 0) process:

$$
X_t=(1-L)^{-d}\varepsilon_t,
\qquad \varepsilon_t\sim\mathcal N(0,1).
$$

The fractional integration weights obey

$$
v_0=1,\qquad v_k=v_{k-1}\frac{d+k-1}{k}.
$$

The seeded simulation generates `SYNTHETIC_N + SYNTHETIC_BURN` observations via causal filtering and discards the burn-in period. Default values are 2,000 retained observations, 500 burn-in observations, $d=0.4$, and seed 42.

The synthetic setting enables controlled assessment of the fixed-d and estimated-d methods against the oracle-H kernel with known generating memory. It is a single-channel experiment per run. Recovery of the true parameter can be assessed using DFA records, but the current output does not export fitted joint-CSS parameters.

## 8. Installation and execution

Python 3.9 or newer is required by the use of `argparse.BooleanOptionalAction` and NumPy's sliding-window utilities require a compatible NumPy version. Use a mutually compatible set of scientific Python packages.

```bash
python -m pip install numpy scipy pandas statsmodels scikit-learn
# Optional seasonal baseline:
python -m pip install tbats
```

The examples assume the script retains its original name, `ARFIMA_e.py`; substitute its actual filename if different.

Run the configured default benchmark:

```bash
python ARFIMA_e.py
```

Inspect command-line options:

```bash
python ARFIMA_e.py --help
```

### 8.1 Real-data experiment

```bash
python ARFIMA_e.py \
  --mode npy \
  --npy-path sp_2016_2d_arfima_vol.npy \
  --channels 0,1,2 --array-layout CT \
  --train-frac 0.95 --horizons 20,30,40,50 \
  --fit-window 500 --rolling-step 10 --max-origins 40 \
  --m-frac 500 --kernel-m 400 \
  --output-dir results_real
```

### 8.2 Synthetic experiment

```bash
python ARFIMA_e.py \
  --mode synthetic --synthetic-n 2000 --synthetic-burn 500 \
  --d-true 0.4 --seed 42 \
  --output-dir results_synthetic
```

### 8.3 Chronological validation and test experiments

Use a chronological validation partition for configuration selection, then assess the selected configuration on a separate test partition. Verify the origin boundary condition in Section 11 for the actual series length. The example uses an 80/10/10 split, rather than the default 95/5 train/test split.

```bash
python ARFIMA_e.py --train-frac 0.8 --val-frac 0.1 \
  --eval-split validation --horizons 20,30,40,50 --rolling-step 11 \
  --output-dir results_validation

python ARFIMA_e.py --train-frac 0.8 --val-frac 0.1 \
  --eval-split test --horizons 20,30,40,50 --rolling-step 11 \
  --output-dir results_test
```

### 8.4 Seasonal comparison (example: hourly data with daily seasonality)

```bash
python ARFIMA_e.py --enable-sarima --sarima-period 24 \
  --sarima-maxiter 100 --enable-tbats --seasonal-periods 24 \
  --output-dir results_seasonal
```

Use only seasonal periods supported by the dataset's sampling frequency. Without TBATS installed, its forecast failures are recorded as unavailable when the benchmark includes it.

### 8.5 Selected methods and sensitivity analysis

```bash
python ARFIMA_e.py \
  --methods 'Naive,ARMA(AIC grid),ARFIMA fixed-d,ARFIMA estimated-d(DFA)' \
  --m-frac 250 --kernel-m 200 --fit-window 500 \
  --arma-p-max 2 --arma-q-max 2 \
  --output-dir results_sensitivity
```

The CLI allows independent variation of memory truncation, kernel length, ARMA search bounds, fitting window length, evaluation stride, horizons, and optimizer budgets. The DFA clipping range is fixed in the source, not exposed as a CLI option. Use distinct output directories for separate runs.

## 9. Command-line configuration reference

| Purpose | Options |
|---|---|
| Data | `--mode`, `--npy-path`, `--channels`, `--array-layout`, `--max-time-points` (0 = no cap) |
| Preprocessing | `--normalize` / `--no-normalize`, `--preprocess-logret`, `--preprocess-abs-logvol` |
| Splits | `--train-frac`, `--val-frac`, `--eval-split` |
| Forecast evaluation | `--horizons`, `--rolling-step`, `--max-origins` (0 = all), `--fit-window` (0 = expanding) |
| ARMA | `--arma-p-max`, `--arma-q-max` |
| Long memory | `--m-frac`, `--kernel-m`, `--d-true`, `--joint-p`, `--joint-q`, `--joint-maxiter` |
| Seasonality | `--enable-sarima` / `--no-enable-sarima`, `--sarima-period`, `--sarima-maxiter`, `--enable-tbats` / `--no-enable-tbats`, `--seasonal-periods` |
| Machine learning | `--enable-ml` / `--no-enable-ml`, `--ml-lags`, `--ml-max-train-samples` |
| Synthetic data | `--synthetic-n`, `--synthetic-burn`, `--seed` |
| Method selection | `--methods` (default `all`; otherwise comma-separated exact names, subject to Section 11) |
| Output | `--output-dir`, `--verbose` / `--no-verbose` |

The joint order arguments are currently constrained to 1 and 1. Horizon, channel, and seasonal-period lists use comma-separated integers. The evaluation partitions must have sufficient observations for the largest horizon and models' minimum history requirements.

## 10. Output files and reproducibility

| File | Description |
|---|---|
| `per_channel_scores.csv` | RMSE, MAE, MASE, number of valid origins, and failures for every evaluated channel, method, and horizon |
| `aggregated_scores.csv` | Cross-channel metric means and sample standard deviations; valid channel counts, total valid origins, and total failures per method and horizon |
| `per_origin_scores.csv` | Channel, origin, method, horizon, validity indicator, runtime, cumulative MSE, and MAE |
| `hurst_by_origin.csv` | Origin-level DFA estimates of H and d, and clipping indicator, when DFA-based ARFIMA is selected |
| `origins_channel_<c>.csv` | Selected rolling-origin indices for channel c |
| `run_config.json` | Effective experiment configuration, including selected method names |

Output channel identifiers are positions in the selected array, not necessarily original dataset channel indices. The original selection is retained in `run_config.json`.

Origin-level runtime measures the combined fitting and maximum-horizon prediction time for one method at one origin. The same elapsed time is repeated for all horizons; do not sum runtime across horizon rows without deduplicating by channel, origin, and method.

The seed controls synthetic Gaussian innovations and randomized ML estimators. For reproducible reporting, retain the output configuration and original data, record the script revision, Python and library versions, and hardware, and use separate output directories for independent runs. The same directory may overwrite previous output files. The code does not guarantee bitwise reproducibility across environments.

## 11. Limitations and implementation notes

- Fractional differencing uses finite truncation; its approximation quality depends on window length, memory parameter, and truncation lag.
- DFA may be sensitive to short windows, trends, regime changes, and seasonal structure. Its use does not establish genuine fractional integration.
- The joint ARFIMA estimator is approximate conditional likelihood, not exact likelihood; finite-objective solutions may be retained even without optimizer convergence.
- ARFIMA two-stage and joint-CSS models currently use order (1, 1); higher ARMA search bounds affect only the standalone ARMA baseline.
- All methods forecast channels independently and use recursive multi-step prediction, where applicable. There is no joint spatial model, prediction interval estimation, or formal significance test in the script.
- Accuracy estimates based on different successful-origin subsets require careful interpretation. Cross-channel standard deviations are descriptive rather than confidence intervals.
- Seasonal forecasts depend on justified periods, adequate history, and sufficient fitting iterations.
- **Origin-boundary caveat:** the supplied code constructs origins with `range(start, stop-hmax+1, ROLLING_STEP)`. If the stride reaches `stop-hmax`, that origin would require a target at index `stop` (outside the chosen partition). It may cause a shape mismatch at the end of the series or validation leakage into test. The intended exclusive upper bound is `stop-hmax` (in both origin-list constructions). Until corrected, check origin indices for each run so that `t + hmax < stop`.
- **Method-selection caveat:** the current comma-separated `--methods` parser cannot select exact names containing embedded commas, including `ARFIMA joint-CSS(1,d,1)` and TBATS with multiple periods. Use `--methods all` to include these methods, or revise the CLI parser before selecting them individually.

## 12. Research use cases

- Fractional long-memory forecasting and Hurst-based model validation.
- Financial price, return, and absolute log-return analysis.
- Environmental and geophysical forecasting, including pressure, wind speed, and temperature measurements.
- Comparisons of fractional models against statistical, seasonal, and nonlinear autoregressors.
- Reproducible multi-horizon and cross-channel benchmarking.
- Sensitivity studies of fractional truncation, rolling-window length, ARMA selection bounds, and fitting budgets.
- Comparative evaluation of fixed, DFA-estimated, and jointly estimated fractional memory.

## 13. Author

**Ali Forootani**

Email: aliforootani@ieee.org · forootani@gea.mpg.de

2026

















#########################################################################
#########################################################################




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

$\text{Cov}(X_t, X_{t+k}) \sim k^{2H-2}$

where ( H \in (0,1) ) is the **Hurst exponent**.

ARFIMA models extend ARIMA by allowing fractional integration:

$(1 - L)^d X_t = \varepsilon_t$

where:

$d = H - 0.5$

This project benchmarks:

* Fixed fractional memory models
* Data-driven memory estimation (via DFA)
* Classical short-memory alternatives

under controlled and real-data settings.

---

## 2. Methodology

### 2.1 Fractional Operators

Fractional differencing:

$(1 - L)^d = \sum_{k=0}^{\infty} w_k L^k$

Weights computed recursively:

$w_k = w_{k-1} \frac{k-1-d}{k}$

Fractional integration:

$(1 - L)^{-d}$

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

$F(s) \sim s^H$

Slope of log-log regression provides ( H ).

Memory parameter:

$d = \text{clip}(H - 0.5, -0.49, 0.49)$

---

### 2.4 Kernel-Based Long-Memory Baseline

Power-law kernel:

$w_k \propto k^{-\eta}$

with

$\eta = H + \frac{1}{2}$

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

$| \Delta \log X_t |$

### Log-Returns

$\Delta \log X_t$

### Per-Channel Standardization

$\frac{X - \mu}{\sigma}$

---

## 7. Performance Metrics

For each horizon ( h ):

* RMSE
* MAE
* MASE

MASE defined as:

$\text{MASE} = \frac{\text{MAE}}{\text{MAE of naive one-step}}$

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
python ARFIMA_e.py
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

