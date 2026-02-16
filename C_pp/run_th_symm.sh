#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults (override via env)
M=${M:-200}
R_LIST=${R_LIST:-"1 3 5 "}
SIGN=${SIGN:-"+"}
GAMMAS=${GAMMAS:-" 0.5 1.0 2.0 4.0"}
SIZES=${SIZES:-800}
TIMES=${TIMES:-"250"}
METHOD=${METHOD:-"rk4"}
RK_STEPS=${RK_STEPS:-900}
RK_TOL=${RK_TOL:-"1e-7"}
RK_ADAPT=${RK_ADAPT:-0}
BETA_L=${BETA_L:-1.0}
BETA_R=${BETA_R:-1.0}
L=${L:-"300"}
NJOBS=${NJOBS:-4 }
OUTDIR=${OUTDIR:-$(pwd)}

for R in $R_LIST; do

  python3 main_dynamics_th_rsym.py \
    --r "$R" \
    --sign "$SIGN" \
    --sizes $SIZES \
    --times $TIMES \
    --gammas $GAMMAS \
    --betaL "$BETA_L" \
    --betaR "$BETA_R" \
    --l     $L \
    --method "$METHOD" \
    --rk-steps "$RK_STEPS" \
    $( [ "$RK_ADAPT" != "0" ] && printf -- "--rk-adapt " ) \
    --rk-tol "$RK_TOL" \
    --n-jobs "$NJOBS" \
    --outdir "$OUTDIR"
done
