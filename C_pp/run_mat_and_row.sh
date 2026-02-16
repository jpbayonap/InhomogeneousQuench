#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults (override via env)
R_LIST=${R_LIST:-"1 3 5"}
SIGN=${SIGN:-"+"}
GAMMAS=${GAMMAS:-"0.5 1.0"}
SIZES=${SIZES:-"1000"}
TIME=${TIME:-"200"}
BETA=${BETA:-"1.0"}
BETA_L=${BETA_L:-"10.0"}
BETA_R=${BETA_R:-"1.0"}
# A_OFFSET is the left-region length: A=(center-A_OFFSET, center]
A_OFFSET=$SIZES
# B_OFFSET is the right-region length: B = [center+1, center+1+B_OFFSET)
B_OFFSET=${B_OFFSET:-"1 5 10 50 100 500 1000"}

METHOD=${METHOD:-"rk4"}
RK_STEPS=${RK_STEPS:-800}
RK_TOL=${RK_TOL:-"1e-7"}
RK_ADAPT=${RK_ADAPT:-0}
NJOBS=${NJOBS:-4}
OUTDIR=${OUTDIR:-"$(pwd)"}

for R in $R_LIST; do
  python3 ./main_dynamics_row.py \
    --r "$R" \
    --sign "$SIGN" \
    --sizes $SIZES \
    --time "$TIME" \
    --gammas $GAMMAS \
    --beta "$BETA" \
    $( [ -n "$BETA_L" ] && printf -- "--betaL %s " "$BETA_L" ) \
    $( [ -n "$BETA_R" ] && printf -- "--betaR %s " "$BETA_R" ) \
    --a-offset "$A_OFFSET" \
    --b-offset $B_OFFSET \
    --method "$METHOD" \
    --rk-steps "$RK_STEPS" \
    $( [ "$RK_ADAPT" != "0" ] && printf -- "--rk-adapt " ) \
    --rk-tol "$RK_TOL" \
    --n-jobs "$NJOBS" \
    --outdir "$OUTDIR"
done
