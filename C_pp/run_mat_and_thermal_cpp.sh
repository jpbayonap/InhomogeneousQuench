#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults (override via env)
SIZES=${SIZES:-"600"}      # half-chain sizes (L); N=2L
TIMES=${TIMES:-"200"}      # times
R_LIST=${R_LIST:-"0 2 4"}  # lattice r values
SIGN=${SIGN:-"+"}
BETAL=${BETAL:-0.0}
BETAR=${BETAR:-1.0}
RK_STEPS=${RK_STEPS:-600}
GAMMAS=${GAMMAS:-"2.0"}
OUTDIR=${OUTDIR:-$(pwd)}

echo "Compiling main_dynamics_thermal.cpp..."
g++ -std=c++17 -O2 main_dynamics_thermal.cpp -I/opt/homebrew/include/eigen3 -o main_dynamics_thermal

CSV_DIR="${OUTDIR}/GHD_THERM_CSV_CPP"
PNG_DIR="${OUTDIR}/GHD_THERM_PNG_CPP"
mkdir -p "$CSV_DIR" "$PNG_DIR"

for L in $SIZES; do
  for TVAL in $TIMES; do
    for R in $R_LIST; do
      echo "Running C++ thermal: L=$L N=$((2*L)) r=$R sign=$SIGN betaL=$BETAL betaR=$BETAR T=$TVAL rk_steps=$RK_STEPS gammas=$GAMMAS"
      ./main_dynamics_thermal "$L" "$R" "$SIGN" "$BETAL" "$BETAR" "$TVAL" "$OUTDIR" "$RK_STEPS" $GAMMAS
    done
  done
done

for csv in "$CSV_DIR"/*.csv; do
  [ -f "$csv" ] || continue
  base=$(basename "$csv" .csv)
  python3 plot_thermal_cpp.py --csv "$csv" --out "$PNG_DIR/${base}.png"
done
