#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults (override via env)
SIZES=${SIZES:-"400"}
TIMES=${TIMES:-"100"}
R_LIST=${R_LIST:-"0 1 2 3 4 5"}

SIGN=${SIGN:-"+"}
BETA_R=${BETA_R:-1.0}
BETA_L=${BETA_L:-0.0}
OUTDIR=${OUTDIR:-.}
N_JOBS=${N_JOBS:-1}
ZMIN=${ZMIN:-""}
ZMAX=${ZMAX:-""}

echo "Running unitary thermal script..."
for R in $R_LIST; do

  cmd=(python3 main_dynamics_thermal_unitary.py
    --sizes $SIZES
    --times $TIMES
    --r $R
    --sign $SIGN
    --betaR $BETA_R
    --betaL $BETA_L
    --outdir "$OUTDIR"
    --n-jobs $N_JOBS
  )
  if [ -n "$ZMIN" ] && [ -n "$ZMAX" ]; then
    cmd+=(--zmin "$ZMIN" --zmax "$ZMAX")
  fi
  echo "${cmd[@]}"
  "${cmd[@]}"
done
