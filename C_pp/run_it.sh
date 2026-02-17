#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
START_EPOCH=$(date +%s)
echo "run_it.sh started at $(date -Iseconds)"

# Keep BLAS single-threaded; parallelism is handled by joblib (--n-jobs).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export GOTO_NUM_THREADS=${GOTO_NUM_THREADS:-1}

# Defaults (override via env)
R_LIST=${R_LIST:-"1"}
SIGN=${SIGN:-"+"}
GAMMAS=${GAMMAS:-"1.0"}
SIZES=${SIZES:-"1000"}
TIMES=${TIMES:-"200 400 600 800 900"}
BETA=${BETA:-"1.0"}
BETA_L=${BETA_L:-""}
BETA_R=${BETA_R:-""}

# S_OFFSET is the right and left region length:  A=[center-S_OFFSET, center], B = [center+1, center+1+S_OFFSET]
S_OFFSET=${S_OFFSET:-"1000"}

METHOD=${METHOD:-"rk4"}
RK_STEPS=${RK_STEPS:-800}
RK_DT_MAX=${RK_DT_MAX:-"0.25"}
RK_TOL=${RK_TOL:-"1e-7"}
RK_ADAPT=${RK_ADAPT:-0}
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
  NJOBS_DEFAULT="${SLURM_CPUS_PER_TASK}"
else
  NJOBS_DEFAULT="5"
fi
NJOBS=${NJOBS:-"$NJOBS_DEFAULT"}
OUTDIR=${OUTDIR:-"$(pwd)"}
PYTHON_BIN=${PYTHON_BIN:-python3}

echo "Using PYTHON_BIN=$PYTHON_BIN, NJOBS=$NJOBS, OUTDIR=$OUTDIR"
echo "TIMES=$TIMES | S_OFFSET=$S_OFFSET | GAMMAS=$GAMMAS"

for R in $R_LIST; do
  "$PYTHON_BIN" ./main_dynamics_it.py \
    --r "$R" \
    --sign "$SIGN" \
    --sizes $SIZES \
    --times $TIMES \
    --gammas $GAMMAS \
    --beta "$BETA" \
    $( [ -n "$BETA_L" ] && printf -- "--betaL %s " "$BETA_L" ) \
    $( [ -n "$BETA_R" ] && printf -- "--betaR %s " "$BETA_R" ) \
    --s-offset $S_OFFSET \
    --method "$METHOD" \
    --rk-steps "$RK_STEPS" \
    $( [ -n "$RK_DT_MAX" ] && printf -- "--rk-dt-max %s " "$RK_DT_MAX" ) \
    $( [ "$RK_ADAPT" != "0" ] && printf -- "--rk-adapt " ) \
    --rk-tol "$RK_TOL" \
    --n-jobs "$NJOBS" \
    --outdir "$OUTDIR"
done

END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
echo "run_it.sh finished at $(date -Iseconds) | elapsed=${ELAPSED}s"
