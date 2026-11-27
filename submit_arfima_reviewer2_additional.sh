#!/bin/bash -l
#SBATCH --job-name=arfima_R2_extra
#SBATCH --partition=s.geany
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=168:00:00
#SBATCH --array=0-90%60
#SBATCH --output=/geany/fs/scratch/alfo/hurst_exponent_arfima/logs/%x-%A_%a.out
#SBATCH --error=/geany/fs/scratch/alfo/hurst_exponent_arfima/logs/%x-%A_%a.err
# Launch on login node: bash submit_arfima_reviewer2_additional.sh
# Default study: 6 persistence levels x 3 sample sizes x 5 seeds = 90 synthetic jobs.
# Example: STUDY=synthetic ARRAY_CONCURRENCY=8 bash submit_arfima_reviewer2_additional.sh
# Optional focused real-data campaign: STUDY=real bash submit_arfima_reviewer2_additional.sh
# Note: This CLI does NOT implement forecast intervals, DM tests, or DFA scale tuning.
set -Eeuo pipefail

STUDY="${STUDY:-real}"  # synthetic | real
DATA_DIR="${DATA_DIR:-/geany/fs/scratch/alfo/hurst_exponent_arfima}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY_SCRIPT="${PY_SCRIPT:-$WORKDIR/ARFIMA_extended_reviewer_benchmark_cli.py}"
VENV="${VENV:-/u/alfo/dnn_env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_DIR/reviewer2_additional_2026}"
LOG_DIR="$DATA_DIR/logs"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-8}"
HORIZONS="${HORIZONS:-20,30,40,50}"
TRAIN_FRAC="${TRAIN_FRAC:-0.95}"
ROLLING_STEP="${ROLLING_STEP:-5}"
MAX_ORIGINS="${MAX_ORIGINS:-0}" # all eligible origins; same origins for every model
M_FRAC="${M_FRAC:-400}"
KERNEL_M="${KERNEL_M:-400}"
ARMA_P_MAX="${ARMA_P_MAX:-2}"
ARMA_Q_MAX="${ARMA_Q_MAX:-2}"
JOINT_MAXITER="${JOINT_MAXITER:-100}"
SYNTHETIC_BURN="${SYNTHETIC_BURN:-1000}"

# Sensitivity to true persistence (d = H - 0.5), length, Monte Carlo seed.
D_VALUES_CSV="${D_VALUES_CSV:--0.20,0.00,0.10,0.25,0.35,0.40}"
N_VALUES_CSV="${N_VALUES_CSV:-2000,3000,5000}"
SEEDS_CSV="${SEEDS_CSV:-12,17,44,122,1254}"
# Focused real-data reproducibility run: avoid repeating all existing settings.
REAL_DATASETS_CSV="${REAL_DATASETS_CSV:-df_cleaned_numeric_2_arfima_vol.npy,wind_speed_2016_2d_arfima_vol.npy,t2m_2016_2d_arfima_vol.npy,sp_2016_2d_arfima_vol.npy,crypto_coins_2d_arfima_vol.npy}"
REAL_FIT_WINDOWS_CSV="${REAL_FIT_WINDOWS_CSV:-300,500}"
REAL_M_FRACS_CSV="${REAL_M_FRACS_CSV:-200,500}"
REAL_D_FIXED="${REAL_D_FIXED:-0.2}" # NOT a true parameter for observational data
REAL_SARIMA_PERIOD="${REAL_SARIMA_PERIOD:-0}" # 0 disables SARIMA (period must be justified)

IFS=',' read -r -a D_VALUES <<< "$D_VALUES_CSV"
IFS=',' read -r -a N_VALUES <<< "$N_VALUES_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a REAL_DATASETS <<< "$REAL_DATASETS_CSV"
IFS=',' read -r -a REAL_FIT_WINDOWS <<< "$REAL_FIT_WINDOWS_CSV"
IFS=',' read -r -a REAL_M_FRACS <<< "$REAL_M_FRACS_CSV"
case "$STUDY" in
 synthetic) TOTAL=$(( ${#D_VALUES[@]} * ${#N_VALUES[@]} * ${#SEEDS[@]} ));;
 real) TOTAL=$(( ${#REAL_DATASETS[@]} * ${#REAL_FIT_WINDOWS[@]} * ${#REAL_M_FRACS[@]} ));;
 *) echo 'STUDY must be synthetic or real' >&2; exit 2;;
esac
((TOTAL>0)) || { echo 'Empty experiment grid' >&2; exit 2; }
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
 [[ -s "$PY_SCRIPT" ]] || { echo "Missing CLI: $PY_SCRIPT" >&2; exit 2; }
 [[ "$ARRAY_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || exit 2
 if [[ "$STUDY" == real ]]; then
  for f in "${REAL_DATASETS[@]}"; do
   [[ -s "$DATA_DIR/$f" ]] || { echo "Missing dataset $f" >&2; exit 2; }
  done
 fi
 mkdir -p "$LOG_DIR" "$OUTPUT_ROOT"
 echo "STUDY=$STUDY jobs=$TOTAL concurrency=$ARRAY_CONCURRENCY"
 echo "Launching with 95% training, 5% testing, horizons=$HORIZONS"
 export STUDY DATA_DIR WORKDIR PY_SCRIPT VENV OUTPUT_ROOT LOG_DIR ARRAY_CONCURRENCY
 export HORIZONS TRAIN_FRAC ROLLING_STEP MAX_ORIGINS M_FRAC KERNEL_M ARMA_P_MAX ARMA_Q_MAX JOINT_MAXITER SYNTHETIC_BURN
 export D_VALUES_CSV N_VALUES_CSV SEEDS_CSV REAL_DATASETS_CSV REAL_FIT_WINDOWS_CSV REAL_M_FRACS_CSV REAL_D_FIXED REAL_SARIMA_PERIOD
 exec sbatch --export=ALL --array="0-$((TOTAL-1))%$ARRAY_CONCURRENCY" "$0"
fi

id="$SLURM_ARRAY_TASK_ID"
((id>=0 && id<TOTAL)) || exit 2
if [[ "$STUDY" == synthetic ]]; then
  k=$id
  seed="${SEEDS[$((k % ${#SEEDS[@]}))]}"; k=$((k/${#SEEDS[@]}))
  n="${N_VALUES[$((k % ${#N_VALUES[@]}))]}"; k=$((k/${#N_VALUES[@]}))
  d="${D_VALUES[$k]}"
  dtag="${d//-/neg}"; dtag="${dtag//./p}"
  out="$OUTPUT_ROOT/synthetic/d_${dtag}/N_${n}/seed_${seed}"
  args=(--mode synthetic --synthetic-n "$n" --synthetic-burn "$SYNTHETIC_BURN"
        --d-true "$d" --seed "$seed" --fit-window 0 --no-enable-sarima
        --no-enable-tbats --no-enable-ml)
else
  k=$id
  mf="${REAL_M_FRACS[$((k % ${#REAL_M_FRACS[@]}))]}"; k=$((k/${#REAL_M_FRACS[@]}))
  fw="${REAL_FIT_WINDOWS[$((k % ${#REAL_FIT_WINDOWS[@]}))]}"; k=$((k/${#REAL_FIT_WINDOWS[@]}))
  dataset="${REAL_DATASETS[$k]}"
  case "$dataset" in
   df_cleaned_numeric_2_arfima_vol.npy) channels=0,1,2,3,4,5;;
   crypto_coins_2d_arfima_vol.npy) channels=0,1;;
   wind_speed_2016_2d_arfima_vol.npy|t2m_2016_2d_arfima_vol.npy|sp_2016_2d_arfima_vol.npy) channels=0,1,2;;
   *) echo "Unknown dataset $dataset" >&2; exit 2;;
  esac
  out="$OUTPUT_ROOT/real/${dataset%.npy}/fit_${fw}_mfrac_${mf}"
  args=(--mode npy --npy-path "$DATA_DIR/$dataset" --channels "$channels"
        --array-layout CT --fit-window "$fw" --m-frac "$mf"
        --d-true "$REAL_D_FIXED" --seed 42 --no-enable-tbats --no-enable-ml)
  if ((REAL_SARIMA_PERIOD>=2)); then
    args+=(--enable-sarima --sarima-period "$REAL_SARIMA_PERIOD")
  else
    args+=(--no-enable-sarima)
  fi
fi
mkdir -p "$out"
[[ ! -s "$out/COMPLETED.ok" ]] || { echo "Already completed: $out"; exit 0; }
module purge
module load gcc/14 || true
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# No stale forecast reuse; the Python CLI refits at EACH evaluation origin.
# '--methods all' avoids parser ambiguity with commas in ARFIMA joint-CSS name.
srun --cpu-bind=cores python -u "$PY_SCRIPT" \
  "${args[@]}" --train-frac "$TRAIN_FRAC" --val-frac 0 --eval-split test \
  --horizons "$HORIZONS" --rolling-step "$ROLLING_STEP" --max-origins "$MAX_ORIGINS" \
  --m-frac "${mf:-$M_FRAC}" --kernel-m "$KERNEL_M" \
  --arma-p-max "$ARMA_P_MAX" --arma-q-max "$ARMA_Q_MAX" \
  --joint-p 1 --joint-q 1 --joint-maxiter "$JOINT_MAXITER" \
  --methods all --output-dir "$out" > >(tee "$out/run.log") 2> >(tee "$out/run.err" >&2)
for file in run_config.json aggregated_scores.csv per_channel_scores.csv per_origin_scores.csv hurst_by_origin.csv; do
  [[ -s "$out/$file" ]] || { echo "Missing $out/$file" >&2; exit 1; }
done
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$out/COMPLETED.ok"
echo "COMPLETED: $out"
