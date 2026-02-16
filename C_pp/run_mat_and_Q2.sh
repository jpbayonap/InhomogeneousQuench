#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults (override via env)
R_LIST=${R_LIST:-"0 1 2 3 4"}
SIGN=${SIGN:-"-"}
GAMMAS=${GAMMAS:-"0.0 1.0 4.0"}
SIZES=${SIZES:-"900"}
TIMES=${TIMES:-"250"}
METHOD=${METHOD:-"rk4"}
RK_STEPS=${RK_STEPS:-650}
RK_TOL=${RK_TOL:-"1e-7"}
RK_ADAPT=${RK_ADAPT:-0}
BETA_L=${BETA_L:-0.0}  # left is vacuum by default
BETA_R=${BETA_R:-1.0}
NJOBS=${NJOBS:-3}
OUTDIR=${OUTDIR:-$(pwd)}

for R in $R_LIST; do
  python3 main_dynamics_Q2.py \
    --r "$R" \
    --sign "$SIGN" \
    --sizes $SIZES \
    --times $TIMES \
    --gammas $GAMMAS \
    --betaL "$BETA_L" \
    --betaR "$BETA_R" \
    --method "$METHOD" \
    --rk-steps "$RK_STEPS" \
    $( [ "$RK_ADAPT" != "0" ] && printf -- "--rk-adapt " ) \
    --rk-tol "$RK_TOL" \
    --n-jobs "$NJOBS" \
    --outdir "$OUTDIR"
done
