#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults (override via env)
R_LIST=${R_LIST:-"1 3"}
SIGN=${SIGN:-"-"}
GAMMAS=${GAMMAS:-"1.0,2.0,4.0"}
SIZES=${SIZES:-"900"}
TIMES=${TIMES:-"250"}
METHOD=${METHOD:-"rk4"}
RK_STEPS=${RK_STEPS:-900}
RK_TOL=${RK_TOL:-"1e-7"}
RK_ADAPT=${RK_ADAPT:-0}
L=${L:-"900"}
NJOBS=${NJOBS:-4}
OUTDIR=${OUTDIR:-"$(pwd)"}

# Compile C++ solver

echo "Compiling main_dynamics.cpp..."
g++ -std=c++17 -O2 -pthread main_dynamics.cpp -I/opt/homebrew/include/eigen3 -o main_dynamics

if [[ "$METHOD" == "rk4" ]]; then
  USE_RK4=1
else
  USE_RK4=0
fi

for R in $R_LIST; do
  ./main_dynamics "$R" "$SIGN" "$USE_RK4" "$RK_ADAPT" "$RK_STEPS" "$RK_TOL" \
  "$SIZES" "$TIMES" "$GAMMAS" "$L" "$OUTDIR"
done
        