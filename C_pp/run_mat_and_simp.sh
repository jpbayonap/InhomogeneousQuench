#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults (override via env)
R_LIST=${R_LIST:-" 2"}
SIGN=${SIGN:-"+"}
GAMMAS=${GAMMAS:-"0.5 1.0"}
SIZES=${SIZES:-"1000"}
TIMES=${TIMES:-"200"}
A_OFFSET=${A_OFFSET:-"0"}
B_OFFSET=${B_OFFSET:-"1"}
METHOD=${METHOD:-"rk4"}
RK_STEPS=${RK_STEPS:-800}
RK_TOL=${RK_TOL:-"1e-7"}
RK_ADAPT=${RK_ADAPT:-0}
NJOBS=${NJOBS:-2}
OUTDIR=${OUTDIR:-"$(pwd)"}

for R in $R_LIST; do
  python3 ./main_dynamics_simp.py \
    --r "$R" \
    --sign "$SIGN" \
    --sizes $SIZES \
    --times $TIMES \
    --gammas $GAMMAS \
    --a-offset "$A_OFFSET" \
    --b-offset "$B_OFFSET" \
    --method "$METHOD" \
    --rk-steps "$RK_STEPS" \
    $( [ "$RK_ADAPT" != "0" ] && printf -- "--rk-adapt " ) \
    --rk-tol "$RK_TOL" \
    --n-jobs "$NJOBS" \
    --outdir "$OUTDIR"
done
