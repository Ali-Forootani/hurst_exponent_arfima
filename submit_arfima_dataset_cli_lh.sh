#!/bin/bash -l
#SBATCH --job-name=arfima_vol_sensitivity
#SBATCH --partition=s.geany
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=168:00:00
#SBATCH --array=0-1919%60
#SBATCH --output=/geany/fs/scratch/alfo/hurst_exponent_arfima/logs/%x-%A_%a.out
#SBATCH --error=/geany/fs/scratch/alfo/hurst_exponent_arfima/logs/%x-%A_%a.err
# Usage (from the directory containing this shell script and Python CLI):
#   bash submit_arfima_dataset_cli.sh
# This login-node call creates the output/log directories and submits a SLURM array.
# To customize a grid: FIT_WINDOWS_CSV=200,500 ROLLING_STEPS_CSV=5,10 bash submit_arfima_dataset_cli.sh
set -Eeuo pipefail

# ============================================================================
# 1. ACTUAL GEANY DATA DIRECTORY: the .npy files are directly inside this path
# ============================================================================
DATA_DIR="${DATA_DIR:-/geany/fs/scratch/alfo/hurst_exponent_arfima}"
DATASETS=(
  "df_cleaned_numeric_2_arfima_vol.npy"    # 6 channels, 0..5
  "wind_speed_2016_2d_arfima_vol.npy"     # 3 channels, 0..2
  "t2m_2016_2d_arfima_vol.npy"            # 3 channels, 0..2
  "sp_2016_2d_arfima_vol.npy"             # 3 channels, 0..2
  "crypto_coins_2d_arfima_vol.npy"        # 2 channels, 0..1
)
ARRAY_LAYOUT="${ARRAY_LAYOUT:-CT}"      # (channels,time); change to TC if needed
MAX_TIME_POINTS="${MAX_TIME_POINTS:-0}" # 0 = entire series
NORMALIZE="${NORMALIZE:-true}"
PREPROCESS_LOGRET="${PREPROCESS_LOGRET:-false}"
PREPROCESS_ABS_LOGVOL="${PREPROCESS_ABS_LOGVOL:-false}"

# ============================================================================
# 2. COMPLETE SENSITIVITY GRID (edit these ranges here, or override via env)
#    D_TRUE in the Python CLI is used as FIXED d for 'ARFIMA fixed-d' even in
#    --mode npy. It is NOT a known true d of any real dataset.
#    ROLLING_STEP and MAX_ORIGINS control evaluation density and sample size,
#    not the model's fit. Changes here compare DIFFERENT origin sets.
# ============================================================================
FIT_WINDOWS_CSV="${FIT_WINDOWS_CSV:-200,300,400,500}"
M_FRACS_CSV="${M_FRACS_CSV:-200,500}"
KERNEL_MS_CSV="${KERNEL_MS_CSV:-200,400}"
ARMA_ORDERS_CSV="${ARMA_ORDERS_CSV:-1:1,2:2}"
ROLLING_STEPS_CSV="${ROLLING_STEPS_CSV:-10,20}"
MAX_ORIGINS_CSV="${MAX_ORIGINS_CSV:-20,40}" # 0 = all eligible origins
D_FIXED_VALUES_CSV="${D_FIXED_VALUES_CSV:-0.0,0.2,0.4}"
SEEDS_CSV="${SEEDS_CSV:-42}"               # expand if enabling stochastic ML

# ============================================================================
# 3. FIXED DATA-SPLIT AND MODEL OPTIONS
# ============================================================================
TRAIN_FRAC=0.95
VAL_FRAC=0.0
EVAL_SPLIT=test
HORIZONS="${HORIZONS:-20,30,40,50}"
JOINT_MAXITER="${JOINT_MAXITER:-100}"
SARIMA_MAXITER="${SARIMA_MAXITER:-100}"
ENABLE_SARIMA="${ENABLE_SARIMA:-true}"
ENABLE_TBATS="${ENABLE_TBATS:-true}"
ENABLE_ML="${ENABLE_ML:-false}"
ML_LAGS="${ML_LAGS:-48}"
ML_MAX_TRAIN_SAMPLES="${ML_MAX_TRAIN_SAMPLES:-2500}"
# IMPORTANT: these are illustrative *assumed* observation-period values.
# Replace with physically/statistically justified seasonality per dataset.
# For no meaningful seasonality, set the periods empty AND disable the methods.
DEFAULT_SARIMA_PERIOD="${DEFAULT_SARIMA_PERIOD:-7}"
DEFAULT_SEASONAL_PERIODS="${DEFAULT_SEASONAL_PERIODS:-7}"
METHODS=all  # Python parser splits named methods on commas, breaking CSS names

# ============================================================================
# 4. CLUSTER ENVIRONMENT
# ============================================================================
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-6}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY_SCRIPT="${PY_SCRIPT:-$WORKDIR/ARFIMA_extended_reviewer_benchmark_cli.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_DIR/real_vol_sensitivity_95train_5test_lh}"
LOG_DIR="$DATA_DIR/logs"
VENV="${VENV:-/u/alfo/dnn_env}"

IFS=',' read -r -a FIT_WINDOWS <<< "$FIT_WINDOWS_CSV"
IFS=',' read -r -a M_FRACS <<< "$M_FRACS_CSV"
IFS=',' read -r -a KERNEL_MS <<< "$KERNEL_MS_CSV"
IFS=',' read -r -a ARMA_ORDERS <<< "$ARMA_ORDERS_CSV"
IFS=',' read -r -a ROLLING_STEPS <<< "$ROLLING_STEPS_CSV"
IFS=',' read -r -a MAX_ORIGINS_LIST <<< "$MAX_ORIGINS_CSV"
IFS=',' read -r -a D_FIXED_VALUES <<< "$D_FIXED_VALUES_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
for name in DATASETS FIT_WINDOWS M_FRACS KERNEL_MS ARMA_ORDERS ROLLING_STEPS MAX_ORIGINS_LIST D_FIXED_VALUES SEEDS; do
  declare -n arr="$name"
  ((${#arr[@]})) || { echo "Empty parameter dimension: $name" >&2; exit 2; }
  unset -n arr
done
TOTAL=$(( ${#DATASETS[@]} * ${#FIT_WINDOWS[@]} * ${#M_FRACS[@]} * ${#KERNEL_MS[@]} * ${#ARMA_ORDERS[@]} * ${#ROLLING_STEPS[@]} * ${#MAX_ORIGINS_LIST[@]} * ${#D_FIXED_VALUES[@]} * ${#SEEDS[@]} ))

# Login-node submitter (never invoke array directly via 'sbatch' for full grid).
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  [[ -s "$PY_SCRIPT" ]] || { echo "Missing Python CLI: $PY_SCRIPT" >&2; exit 2; }
  for f in "${DATASETS[@]}"; do
    [[ -s "$DATA_DIR/$f" ]] || { echo "Missing dataset: $DATA_DIR/$f" >&2; exit 2; }
  done
  [[ "$ARRAY_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid ARRAY_CONCURRENCY' >&2; exit 2; }
  mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"
  echo "DATA_DIR=$DATA_DIR"
  echo "Total jobs: $TOTAL; concurrent: $ARRAY_CONCURRENCY"
  echo "Split: train=0.95 validation=0 test=0.05; methods=$METHODS"
  echo "NOTE: d-fixed varies only ARFIMA fixed-d; rolling settings change test-origin sampling."
  echo "NOTE: SARIMA/TBATS periods must match real seasonal frequencies; default 7 is illustrative."
  export DATA_DIR ARRAY_LAYOUT MAX_TIME_POINTS NORMALIZE PREPROCESS_LOGRET PREPROCESS_ABS_LOGVOL
  export FIT_WINDOWS_CSV M_FRACS_CSV KERNEL_MS_CSV ARMA_ORDERS_CSV
  export ROLLING_STEPS_CSV MAX_ORIGINS_CSV D_FIXED_VALUES_CSV SEEDS_CSV
  export TRAIN_FRAC VAL_FRAC EVAL_SPLIT HORIZONS JOINT_MAXITER SARIMA_MAXITER
  export ENABLE_SARIMA ENABLE_TBATS ENABLE_ML ML_LAGS ML_MAX_TRAIN_SAMPLES
  export DEFAULT_SARIMA_PERIOD DEFAULT_SEASONAL_PERIODS METHODS
  export ARRAY_CONCURRENCY WORKDIR PY_SCRIPT OUTPUT_ROOT VENV
  exec sbatch --export=ALL --array="0-$((TOTAL-1))%$ARRAY_CONCURRENCY" "$0"
fi

# Decode a unique Cartesian-product task index.
TASK_ID="$SLURM_ARRAY_TASK_ID"
((TASK_ID >= 0 && TASK_ID < TOTAL)) || { echo "Out-of-range task index $TASK_ID" >&2; exit 2; }
i=$TASK_ID
idx=$((i % ${#SEEDS[@]}));                 SEED="${SEEDS[$idx]}"; i=$((i / ${#SEEDS[@]}))
idx=$((i % ${#D_FIXED_VALUES[@]}));        D_FIXED="${D_FIXED_VALUES[$idx]}"; i=$((i / ${#D_FIXED_VALUES[@]}))
idx=$((i % ${#MAX_ORIGINS_LIST[@]}));      MAX_ORIGINS="${MAX_ORIGINS_LIST[$idx]}"; i=$((i / ${#MAX_ORIGINS_LIST[@]}))
idx=$((i % ${#ROLLING_STEPS[@]}));         ROLLING_STEP="${ROLLING_STEPS[$idx]}"; i=$((i / ${#ROLLING_STEPS[@]}))
idx=$((i % ${#ARMA_ORDERS[@]}));           ARMA_PAIR="${ARMA_ORDERS[$idx]}"; i=$((i / ${#ARMA_ORDERS[@]}))
idx=$((i % ${#KERNEL_MS[@]}));             KERNEL_M="${KERNEL_MS[$idx]}"; i=$((i / ${#KERNEL_MS[@]}))
idx=$((i % ${#M_FRACS[@]}));               M_FRAC="${M_FRACS[$idx]}"; i=$((i / ${#M_FRACS[@]}))
idx=$((i % ${#FIT_WINDOWS[@]}));           FIT_WINDOW="${FIT_WINDOWS[$idx]}"; i=$((i / ${#FIT_WINDOWS[@]}))
DATASET="${DATASETS[$i]}"
NPY_PATH="$DATA_DIR/$DATASET"
DATASET_NAME="${DATASET%.npy}"
IFS=':' read -r ARMA_P_MAX ARMA_Q_MAX <<< "$ARMA_PAIR"
[[ "$ARMA_P_MAX" =~ ^[0-9]+$ && "$ARMA_Q_MAX" =~ ^[0-9]+$ ]] || { echo "Bad ARMA pair: $ARMA_PAIR" >&2; exit 2; }
[[ "$ROLLING_STEP" =~ ^[1-9][0-9]*$ && "$MAX_ORIGINS" =~ ^[0-9]+$ ]] || { echo 'Bad origin settings' >&2; exit 2; }

# Explicit per-dataset channel selection as requested.
case "$DATASET" in
  df_cleaned_numeric_2_arfima_vol.npy) CHANNELS="0,1,2,3,4,5" ;;
  crypto_coins_2d_arfima_vol.npy)       CHANNELS="0,1" ;;
  wind_speed_2016_2d_arfima_vol.npy|t2m_2016_2d_arfima_vol.npy|sp_2016_2d_arfima_vol.npy)
    CHANNELS="0,1,2" ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac

# Seasonal periods. These may be overridden per dataset if scientifically justified.
SARIMA_PERIOD="$DEFAULT_SARIMA_PERIOD"
SEASONAL_PERIODS="$DEFAULT_SEASONAL_PERIODS"
case "$DATASET" in
  df_cleaned_numeric_2_arfima_vol.npy) : ;;
  wind_speed_2016_2d_arfima_vol.npy) : ;;
  t2m_2016_2d_arfima_vol.npy) : ;;
  sp_2016_2d_arfima_vol.npy) : ;;
  crypto_coins_2d_arfima_vol.npy) : ;;
esac

boolean_flag() {
  local key="$1" value="${2,,}"
  case "$value" in
    true|1|yes) printf '%s' "--$key" ;;
    false|0|no) printf '%s' "--no-$key" ;;
    *) echo "Invalid $key=$2" >&2; exit 2 ;;
  esac
}
SARIMA_FLAG="$(boolean_flag enable-sarima "$ENABLE_SARIMA")"
TBATS_FLAG="$(boolean_flag enable-tbats "$ENABLE_TBATS")"
ML_FLAG="$(boolean_flag enable-ml "$ENABLE_ML")"
NORMALIZE_FLAG="$(boolean_flag normalize "$NORMALIZE")"
PREPROCESS_FLAGS=()
case "${PREPROCESS_LOGRET,,}" in true|1|yes) PREPROCESS_FLAGS+=(--preprocess-logret);; false|0|no) :;; *) exit 2;; esac
case "${PREPROCESS_ABS_LOGVOL,,}" in true|1|yes) PREPROCESS_FLAGS+=(--preprocess-abs-logvol);; false|0|no) :;; *) exit 2;; esac

module purge
module load gcc/14 || true
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_DYNAMIC=FALSE
[[ -s "$PY_SCRIPT" && -s "$NPY_PATH" ]] || { echo "Missing Python script or input: $PY_SCRIPT / $NPY_PATH" >&2; exit 2; }
if [[ "$TBATS_FLAG" == --enable-tbats ]]; then
  python -c 'import tbats' || { echo 'TBATS not installed; pip install tbats (or ENABLE_TBATS=false)' >&2; exit 2; }
fi

D_TAG="${D_FIXED//-/neg}"; D_TAG="${D_TAG//./p}"
OUT_DIR="$OUTPUT_ROOT/$DATASET_NAME/fit_${FIT_WINDOW}_frac_${M_FRAC}_kernel_${KERNEL_M}_arma_${ARMA_P_MAX}_${ARMA_Q_MAX}/step_${ROLLING_STEP}_maxorig_${MAX_ORIGINS}_dfixed_${D_TAG}/seed_${SEED}"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/COMPLETED.ok"
printf 'task=%s file=%s channels=%s fit=%s frac=%s kernel=%s arma=%s step=%s maxorig=%s d_fixed=%s seed=%s\n' \
  "$TASK_ID" "$NPY_PATH" "$CHANNELS" "$FIT_WINDOW" "$M_FRAC" "$KERNEL_M" "$ARMA_PAIR" "$ROLLING_STEP" "$MAX_ORIGINS" "$D_FIXED" "$SEED"

srun --cpu-bind=cores python -u "$PY_SCRIPT" \
  --mode npy --npy-path "$NPY_PATH" --channels "$CHANNELS" \
  --array-layout "$ARRAY_LAYOUT" --max-time-points "$MAX_TIME_POINTS" \
  --train-frac "$TRAIN_FRAC" --val-frac "$VAL_FRAC" --eval-split "$EVAL_SPLIT" \
  --horizons "$HORIZONS" --rolling-step "$ROLLING_STEP" --max-origins "$MAX_ORIGINS" \
  --fit-window "$FIT_WINDOW" --m-frac "$M_FRAC" --kernel-m "$KERNEL_M" \
  --arma-p-max "$ARMA_P_MAX" --arma-q-max "$ARMA_Q_MAX" --joint-p 1 --joint-q 1 \
  --joint-maxiter "$JOINT_MAXITER" --sarima-maxiter "$SARIMA_MAXITER" \
  --sarima-period "$SARIMA_PERIOD" --seasonal-periods "$SEASONAL_PERIODS" \
  --ml-lags "$ML_LAGS" --ml-max-train-samples "$ML_MAX_TRAIN_SAMPLES" \
  --seed "$SEED" --d-true "$D_FIXED" \
  "$NORMALIZE_FLAG" "$SARIMA_FLAG" "$TBATS_FLAG" "$ML_FLAG" \
  "${PREPROCESS_FLAGS[@]}" --methods "$METHODS" --output-dir "$OUT_DIR" \
  > >(tee "$OUT_DIR/run.log") 2> >(tee "$OUT_DIR/run.err" >&2)

for f in run_config.json aggregated_scores.csv per_channel_scores.csv per_origin_scores.csv hurst_by_origin.csv; do
  [[ -s "$OUT_DIR/$f" ]] || { echo "Missing output: $OUT_DIR/$f" >&2; exit 1; }
done
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUT_DIR/COMPLETED.ok"
echo "COMPLETED: $OUT_DIR"
