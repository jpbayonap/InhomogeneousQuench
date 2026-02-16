#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults (override via env)
M=${M:-200}
R_LIST=${R_LIST:-"1 3"}
SIGN=${SIGN:-"-"}
GAMMAS=${GAMMAS:-"1.0 2.0 4.0"}
SIZES=${SIZES:-"900"}
TIMES=${TIMES:-"200"}
METHOD=${METHOD:-"rk4"}
RK_STEPS=${RK_STEPS:-900}
RK_TOL=${RK_TOL:-"1e-7"}
RK_ADAPT=${RK_ADAPT:-0}
L=${L:-"900"}
NJOBS=${NJOBS:-4 }
OUTDIR=${OUTDIR:-$(pwd)}

for R in $R_LIST; do

  python3 main_dynamics_r.py \
    --r "$R" \
    --sign "$SIGN" \
    --sizes $SIZES \
    --times $TIMES \
    --gammas $GAMMAS \
    --l     $L \
    --method "$METHOD" \
    --rk-steps "$RK_STEPS" \
    $( [ "$RK_ADAPT" != "0" ] && printf -- "--rk-adapt " ) \
    --rk-tol "$RK_TOL" \
    --n-jobs "$NJOBS" \
    --outdir "$OUTDIR"
done
