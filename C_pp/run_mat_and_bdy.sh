#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults (override via env)
M=${M:-200}
R_LIST=${R_LIST:-"1 3 5"}
SIGN=${SIGN:-"-"}
GAMMAS=${GAMMAS:-"0 0.1 0.5 1.0"}
SIZES=${SIZES:-"900"}
TIMES=${TIMES:-"200"}
METHOD=${METHOD:-"rk4"}
RK_STEPS=${RK_STEPS:-600}
RK_TOL=${RK_TOL:-"1e-5"}
RK_ADAPT=${RK_ADAPT:-0}
NJOBS=${NJOBS:-2}
OUTDIR=${OUTDIR:-$(pwd)}

# Only run lattice numerics + overlay using existing Mathematica CSVs (no MathKernel call here)
for R in $R_LIST; do
  python3 main_dynamics_BDY.py \
    --r "$R" \
    --sign "$SIGN" \
    --sizes $SIZES \
    --times $TIMES \
    --gammas $GAMMAS \
    --qags-sign "$SIGN" \
    --qags-M "$M" \
    --qags-dir "$OUTDIR" \
    --qags-pattern "GHD_r{r}_sign{sign}_M{M}_gamma{gamma:.6f}test.csv"\
    --method "$METHOD" \
    --rk-steps "$RK_STEPS" \
    $( [ "$RK_ADAPT" != "0" ] && printf -- "--rk-adapt " ) \
    --rk-tol "$RK_TOL" \
    --n-jobs "$NJOBS" \
    --outdir "$OUTDIR"
done
