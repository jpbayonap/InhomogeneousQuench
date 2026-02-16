#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults (override via env)
OUTDIR=${OUTDIR:-"$(pwd)"}
R_LIST=${R_LIST:-"1,3"}
SIGN=${SIGN:-"-"}
GAMMAS=${GAMMAS:-"1.0,2.0,4.0"}
SIZES=${SIZES:-"900"}
TIMES=${TIMES:-"250"}
L=${L:-"800"}
ZMIN=${ZMIN:-"-2.5"}
ZMAX=${ZMAX:-"2.5"}

python3 plot_nell_cpp.py \
  --outdir "$OUTDIR" \
  --r-list "$R_LIST" \
  --sign "$SIGN" \
  --gammas "$GAMMAS" \
  --sizes "$SIZES" \
  --times "$TIMES" \
  --lengths "$L" \
  --zmin "$ZMIN" \
  --zmax "$ZMAX"
