#!/bin/bash -l
#SBATCH --job-name=arfima_R2_robust
#SBATCH --partition=s.geany
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=168:00:00
#SBATCH --output=/geany/fs/scratch/alfo/hurst_exponent_arfima/logs/%x-%A_%a.out
#SBATCH --error=/geany/fs/scratch/alfo/hurst_exponent_arfima/logs/%x-%A_%a.err
# Usage:
#   STUDY=real bash submit_arfima_reviewer_robustness.sh
#   STUDY=synthetic bash submit_arfima_reviewer_robustness.sh
#   STUDY=analyze bash submit_arfima_reviewer_robustness.sh
# No edits to ARFIMA_extended_reviewer_benchmark_cli.py are needed.
set -Eeuo pipefail
STUDY="${STUDY:-real}"
DATA_DIR="${DATA_DIR:-/geany/fs/scratch/alfo/hurst_exponent_arfima}"
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY_SCRIPT="${PY_SCRIPT:-$WORKDIR/ARFIMA_extended_reviewer_benchmark_cli.py}"
VENV="${VENV:-/u/alfo/dnn_env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_DIR/reviewer2_robustness_2026}"
LOG_DIR="$DATA_DIR/logs"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-8}"
HORIZONS="${HORIZONS:-20,30,40,50}"
TRAIN_FRAC="${TRAIN_FRAC:-0.95}"
ROLLING_STEP="${ROLLING_STEP:-5}"
MAX_ORIGINS="${MAX_ORIGINS:-0}"
FIT_WINDOW="${FIT_WINDOW:-500}"
M_FRAC="${M_FRAC:-400}"
KERNEL_M="${KERNEL_M:-400}"
ARMA_P_MAX="${ARMA_P_MAX:-2}"
ARMA_Q_MAX="${ARMA_Q_MAX:-2}"
JOINT_MAXITER="${JOINT_MAXITER:-100}"
SYNTHETIC_BURN="${SYNTHETIC_BURN:-1000}"
D_VALUES_CSV="${D_VALUES_CSV:--0.20,0.00,0.10,0.25,0.35,0.40}"
N_VALUES_CSV="${N_VALUES_CSV:-2000,3000,5000}"
SEEDS_CSV="${SEEDS_CSV:-12,17,44,122,1254}"
REAL_DATASETS_CSV="${REAL_DATASETS_CSV:-df_cleaned_numeric_2_arfima_vol.npy,wind_speed_2016_2d_arfima_vol.npy,t2m_2016_2d_arfima_vol.npy,sp_2016_2d_arfima_vol.npy,crypto_coins_2d_arfima_vol.npy}"
# One-factor-at-a-time tests; each uses IDENTICAL origins, split, ARMA grid.
# Scenarios: label:estimator:dfa_min:dfa_max_divisor:dfa_nscales:d_cap:m_frac:d_shift
SCENARIOS=(
  'baseline:dfa:8:4:12:0.49:400:0'
  'dfa_small_scales:dfa:8:8:12:0.49:400:0'
  'dfa_large_scales:dfa:16:3:16:0.49:400:0'
  'clip_030:dfa:8:4:12:0.30:400:0'
  'clip_040:dfa:8:4:12:0.40:400:0'
  'filter_100:dfa:8:4:12:0.49:100:0'
  'filter_800:dfa:8:4:12:0.49:800:0'
  'gph_estimator:gph:8:4:12:0.49:400:0'
  'd_minus_010:dfa:8:4:12:0.49:400:-0.10'
  'd_plus_010:dfa:8:4:12:0.49:400:0.10'
)
IFS=',' read -r -a D_VALUES <<< "$D_VALUES_CSV"
IFS=',' read -r -a N_VALUES <<< "$N_VALUES_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a REAL_DATASETS <<< "$REAL_DATASETS_CSV"

# Create an auxiliary wrapper. This imports but NEVER edits the original Python CLI.
write_wrapper() {
cat > "$OUTPUT_ROOT/_runtime_wrapper.py" <<'PY'
import os, sys, importlib.util, numpy as np
from pathlib import Path
path = Path(os.environ['PY_SCRIPT']).resolve()
spec = importlib.util.spec_from_file_location('arfima_original_unchanged', path)
cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cli
spec.loader.exec_module(cli)
original_hurst = cli.hurst_dfa
original_joint = cli.fc_arfima_joint_css
estimator = os.environ['ROBUST_ESTIMATOR']
scale_min = int(os.environ['ROBUST_SCALE_MIN'])
scale_divisor = int(os.environ['ROBUST_SCALE_DIV'])
scale_count = int(os.environ['ROBUST_SCALE_COUNT'])
d_cap = float(os.environ['ROBUST_D_CAP'])
d_shift = float(os.environ['ROBUST_D_SHIFT'])

def hurst_custom(values):
    x = np.asarray(values, float)
    if estimator == 'dfa' and scale_min == 8 and scale_divisor == 4 and scale_count == 12:
        return original_hurst(x)  # exact original DFA in baseline
    if len(x) < 64:
        return 0.5
    if estimator == 'gph':
        # Geweke--Porter--Hudak log-periodogram regression: spectral estimate of d.
        z = x - np.mean(x)
        m = max(10, min(len(x)//2-1, int(len(x)**0.6)))
        freqs = 2*np.pi*np.arange(1, m+1)/len(x)
        power = (abs(np.fft.rfft(z))[1:m+1]**2) / len(x)
        good = np.isfinite(power) & (power > 0)
        if good.sum() < 8:
            return 0.5
        feature = -2*np.log(2*np.sin(freqs[good]/2))
        d = np.polyfit(feature, np.log(power[good]), 1)[0]
        return float(np.clip(0.5 + d, 0.01, 0.99))
    profile = np.cumsum(x-x.mean())
    scales = np.unique(np.geomspace(scale_min, max(scale_min+1, len(x)//scale_divisor), scale_count).astype(int))
    Fs, Ss = [], []
    for s in scales:
        n = len(x)//s
        if n < 2: continue
        arr = profile[:n*s].reshape(n, s)
        tt = np.arange(s, dtype=float)
        tc = tt - tt.mean()
        denom = np.dot(tc, tc)
        slopes = ((arr - arr.mean(axis=1, keepdims=True)) @ tc)/denom
        trend = arr.mean(axis=1, keepdims=True) + slopes[:, None]*tc[None, :]
        f = np.sqrt(np.mean((arr-trend)**2))
        if f > 0 and np.isfinite(f):
            Fs.append(f); Ss.append(s)
    return float(np.clip(np.polyfit(np.log(Ss), np.log(Fs), 1)[0], 0.01, 0.99)) if len(Fs)>=4 else .5

# Override only at import-time; no modifications to the original source on disk.
cli.hurst_dfa = hurst_custom

def modified_dfa_forecast(x, h):
    d_raw = cli.hurst_dfa(x) - 0.5
    d = float(np.clip(d_raw + d_shift, -d_cap, d_cap))
    return cli.fc_arfima_pipeline(x, h, d)

cli.fc_arfima_dfa = modified_dfa_forecast

# Keep the joint-CSS baseline's DFA initialization identical across all tests.
def unmodified_joint_css(x, h):
    active = cli.hurst_dfa
    cli.hurst_dfa = original_hurst
    try:
        return original_joint(x, h)
    finally:
        cli.hurst_dfa = active
cli.fc_arfima_joint_css = unmodified_joint_css

# CLI arguments are passed through unchanged.
cli.apply_args(cli.parse_args())
cli.main()
PY
}

write_analysis() {
cat > "$OUTPUT_ROOT/_analyze.py" <<'PY'
import os
from pathlib import Path
import numpy as np
import pandas as pd
root = Path(os.environ['OUTPUT_ROOT'])/'real'
R = np.random.default_rng(20261104)
B = int(os.environ.get('BOOTSTRAP_REPS','1000'))
step = int(os.environ.get('ROLLING_STEP','5'))
method = 'ARFIMA estimated-d(DFA)'

# Paired circular moving-block resampling of forecast-origin MSE; it assesses
# point-forecast performance variability, NOT full first-stage parameter uncertainty.
def paired_ci(a, b, h):
    a = np.asarray(a, float); b = np.asarray(b, float)
    good = np.isfinite(a)&np.isfinite(b)
    a=a[good]; b=b[good]
    n=len(a)
    if n < 4: return (np.nan, np.nan, np.nan, n)
    gain = 100*(1-np.sqrt(np.mean(b))/np.sqrt(np.mean(a))) if np.mean(a)>0 else np.nan
    block = min(n, max(2, int(np.ceil(h/step))))
    sims=[]
    for _ in range(B):
        starts=R.integers(0,n,size=int(np.ceil(n/block)))
        ids=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]
        if np.mean(a[ids])>0:
            sims.append(100*(1-np.sqrt(np.mean(b[ids]))/np.sqrt(np.mean(a[ids]))))
    lo,hi=np.percentile(sims,[2.5,97.5]) if sims else (np.nan,np.nan)
    return gain,lo,hi,n

out=[]; drows=[]
for dataset_dir in sorted(root.glob('*')):
    baseline = dataset_dir/'baseline'/'per_origin_scores.csv'
    if not baseline.exists(): continue
    base = pd.read_csv(baseline)
    keys=['channel','origin','horizon']
    ref = base.loc[(base.name==method)&(base.valid==True),keys+['mse']]
    joint=base.loc[(base.name=='ARFIMA joint-CSS(1,d,1)')&(base.valid==True),keys+['mse']]
    for scenario_dir in sorted(dataset_dir.iterdir()):
        f=scenario_dir/'per_origin_scores.csv'
        if not f.exists() or scenario_dir.name=='baseline': continue
        test=pd.read_csv(f)
        alt=test.loc[(test.name==method)&(test.valid==True),keys+['mse']]
        matched=ref.merge(alt,on=keys,suffixes=('_base','_alt'))
        for (ch,h), g in matched.sort_values('origin').groupby(['channel','horizon']):
            gain,lo,hi,n=paired_ci(g.mse_base,g.mse_alt,h)
            out.append(dict(dataset=dataset_dir.name,scenario=scenario_dir.name,
                            comparison='variant versus baseline DFA',channel=ch,horizon=h,
                            n_paired_origins=n,relative_rmse_gain_percent=gain,
                            ci_low_percent=lo,ci_high_percent=hi))
        hf=scenario_dir/'hurst_by_origin.csv'
        if hf.exists():
            hd=pd.read_csv(hf)
            if len(hd):
                cap=float(os.environ.get('DEFAULT_CAP','0.49'))
                if scenario_dir.name=='clip_030': cap=.30
                if scenario_dir.name=='clip_040': cap=.40
                shift=-.1 if scenario_dir.name=='d_minus_010' else (.1 if scenario_dir.name=='d_plus_010' else 0.)
                drows.append(dict(dataset=dataset_dir.name,scenario=scenario_dir.name,
                                  n=len(hd),d_mean=hd.d_est_DFA.mean(),d_sd=hd.d_est_DFA.std(),
                                  fraction_at_effective_cap=float((abs(hd.d_est_DFA+shift)>=cap).mean()),
                                  note='d_est_DFA column is the H-estimator output, before scenario shift/cap'))
    matched=ref.merge(joint,on=keys,suffixes=('_base','_alt'))
    for (ch,h),g in matched.sort_values('origin').groupby(['channel','horizon']):
        gain,lo,hi,n=paired_ci(g.mse_base,g.mse_alt,h)
        out.append(dict(dataset=dataset_dir.name,scenario='baseline',
                        comparison='joint CSS versus two-stage DFA',channel=ch,horizon=h,
                        n_paired_origins=n,relative_rmse_gain_percent=gain,
                        ci_low_percent=lo,ci_high_percent=hi))
if not out:
    raise SystemExit('No completed matched real-data runs found; run STUDY=real first.')
pd.DataFrame(out).to_csv(Path(os.environ['OUTPUT_ROOT'])/'paired_robustness_cis.csv',index=False)
if drows:
    pd.DataFrame(drows).to_csv(Path(os.environ['OUTPUT_ROOT'])/'hurst_robustness_summary.csv',index=False)
print('Wrote paired_robustness_cis.csv and hurst_robustness_summary.csv')
print('CIs are heuristic origin-block CIs, not a generated-parameter uncertainty correction.')
PY
}

if [[ "$STUDY" == analyze ]]; then
  mkdir -p "$OUTPUT_ROOT"
  write_analysis
  source "$VENV/bin/activate"
  export OUTPUT_ROOT ROLLING_STEP
  python "$OUTPUT_ROOT/_analyze.py"
  exit 0
fi

case "$STUDY" in
 real) TOTAL=$(( ${#REAL_DATASETS[@]} * ${#SCENARIOS[@]} ));;
 synthetic) TOTAL=$(( ${#D_VALUES[@]} * ${#N_VALUES[@]} * ${#SEEDS[@]} ));;
 *) echo 'STUDY must be real, synthetic, or analyze' >&2; exit 2;;
esac
(( TOTAL > 0 )) || exit 2
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  [[ -s "$PY_SCRIPT" ]] || { echo "Missing Python CLI: $PY_SCRIPT" >&2; exit 2; }
  [[ "$ARRAY_CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || exit 2
  if [[ "$STUDY" == real ]]; then
    for f in "${REAL_DATASETS[@]}"; do
      [[ -s "$DATA_DIR/$f" ]] || { echo "Missing input: $DATA_DIR/$f" >&2; exit 2; }
    done
  fi
  mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
  write_wrapper
  echo "$STUDY: $TOTAL jobs; max concurrency $ARRAY_CONCURRENCY"
  echo "Unchanged Python source: $PY_SCRIPT"
  echo "NOTE: scale/clipping and GPH tests are runtime overrides, not built-in CLI features."
  export STUDY DATA_DIR WORKDIR PY_SCRIPT VENV OUTPUT_ROOT LOG_DIR ARRAY_CONCURRENCY
  export HORIZONS TRAIN_FRAC ROLLING_STEP MAX_ORIGINS FIT_WINDOW M_FRAC KERNEL_M
  export ARMA_P_MAX ARMA_Q_MAX JOINT_MAXITER SYNTHETIC_BURN
  export D_VALUES_CSV N_VALUES_CSV SEEDS_CSV REAL_DATASETS_CSV
  exec sbatch --export=ALL --array="0-$((TOTAL-1))%$ARRAY_CONCURRENCY" "$0"
fi

id="$SLURM_ARRAY_TASK_ID"
((id>=0 && id<TOTAL)) || exit 2
ROBUST_ESTIMATOR=dfa ROBUST_SCALE_MIN=8 ROBUST_SCALE_DIV=4 ROBUST_SCALE_COUNT=12
ROBUST_D_CAP=0.49 ROBUST_D_SHIFT=0
if [[ "$STUDY" == real ]]; then
  si=$((id % ${#SCENARIOS[@]}))
  di=$((id / ${#SCENARIOS[@]}))
  dataset="${REAL_DATASETS[$di]}"
  IFS=':' read -r label ROBUST_ESTIMATOR ROBUST_SCALE_MIN ROBUST_SCALE_DIV ROBUST_SCALE_COUNT ROBUST_D_CAP local_m ROBUST_D_SHIFT <<< "${SCENARIOS[$si]}"
  case "$dataset" in
    df_cleaned_numeric_2_arfima_vol.npy) channels='0,1,2,3,4,5';;
    crypto_coins_2d_arfima_vol.npy) channels='0,1';;
    wind_speed_2016_2d_arfima_vol.npy|t2m_2016_2d_arfima_vol.npy|sp_2016_2d_arfima_vol.npy) channels='0,1,2';;
    *) echo "Unknown dataset: $dataset" >&2; exit 2;;
  esac
  out="$OUTPUT_ROOT/real/${dataset%.npy}/$label"
  args=(--mode npy --npy-path "$DATA_DIR/$dataset" --channels "$channels"
        --array-layout CT --fit-window "$FIT_WINDOW" --m-frac "$local_m"
        --d-true 0.2 --seed 42)
else
  k=$id
  seed="${SEEDS[$((k % ${#SEEDS[@]}))]}"; k=$((k/${#SEEDS[@]}))
  n="${N_VALUES[$((k % ${#N_VALUES[@]}))]}"; k=$((k/${#N_VALUES[@]}))
  d="${D_VALUES[$k]}"
  dtag="${d//-/neg}"; dtag="${dtag//./p}"
  out="$OUTPUT_ROOT/synthetic/d_${dtag}/N_${n}/seed_${seed}"
  local_m="$M_FRAC"
  args=(--mode synthetic --synthetic-n "$n" --synthetic-burn "$SYNTHETIC_BURN"
        --d-true "$d" --seed "$seed" --fit-window 0 --m-frac "$local_m")
fi
mkdir -p "$out"
[[ ! -s "$out/COMPLETED.ok" ]] || { echo "Already completed: $out"; exit 0; }
module purge
module load gcc/14 || true
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PY_SCRIPT ROBUST_ESTIMATOR ROBUST_SCALE_MIN ROBUST_SCALE_DIV ROBUST_SCALE_COUNT ROBUST_D_CAP ROBUST_D_SHIFT
printf 'study=%s task=%s out=%s estimator=%s scale=(%s,%s,%s) cap=%s delta=%s m=%s\n' \
  "$STUDY" "$id" "$out" "$ROBUST_ESTIMATOR" "$ROBUST_SCALE_MIN" "$ROBUST_SCALE_DIV" "$ROBUST_SCALE_COUNT" "$ROBUST_D_CAP" "$ROBUST_D_SHIFT" "$local_m"
# All core methods incl. joint CSS; SARIMA/TBATS/ML switched off for this FOCUSED study.
# Historical benchmark results with more methods remain separate.
srun --cpu-bind=cores python -u "$OUTPUT_ROOT/_runtime_wrapper.py" \
  "${args[@]}" --train-frac "$TRAIN_FRAC" --val-frac 0 --eval-split test \
  --horizons "$HORIZONS" --rolling-step "$ROLLING_STEP" --max-origins "$MAX_ORIGINS" \
  --kernel-m "$KERNEL_M" --arma-p-max "$ARMA_P_MAX" --arma-q-max "$ARMA_Q_MAX" \
  --joint-p 1 --joint-q 1 --joint-maxiter "$JOINT_MAXITER" \
  --no-enable-sarima --no-enable-tbats --no-enable-ml \
  --methods all --output-dir "$out" > >(tee "$out/run.log") 2> >(tee "$out/run.err" >&2)
for file in run_config.json aggregated_scores.csv per_channel_scores.csv per_origin_scores.csv hurst_by_origin.csv; do
  [[ -s "$out/$file" ]] || { echo "Missing $out/$file" >&2; exit 1; }
done
printf 'scenario=%s\nestimator=%s\nscale_min=%s\nscale_max_divisor=%s\nscale_count=%s\nd_cap=%s\nd_shift=%s\nfilter_length=%s\n' \
  "${label:-synthetic_baseline}" "$ROBUST_ESTIMATOR" "$ROBUST_SCALE_MIN" "$ROBUST_SCALE_DIV" "$ROBUST_SCALE_COUNT" "$ROBUST_D_CAP" "$ROBUST_D_SHIFT" "$local_m" > "$out/robustness_scenario.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$out/COMPLETED.ok"
echo "COMPLETED: $out"
