#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults (override via env)
M=${M:-800}
R_LIST=${R_LIST:-"1 3 5"}
SIGN=${SIGN:-"-"}
GAMMAS=${GAMMAS:-"0 0.1 0.5 1.0"}
ZMIN=${ZMIN:--2.3}
ZMAX=${ZMAX:-2.3}
MAT_DIR=${MAT_DIR:-.}
PY_DIR=${PY_DIR:-.}

python3 compare_csv_plot.py \
  --M "$M" \
  --r-list $R_LIST \
  --sign "$SIGN" \
  --gammas $GAMMAS \
  --mat-dir "$MAT_DIR" \
  --py-dir "$PY_DIR" \
  --zmin "$ZMIN" \
  --zmax "$ZMAX"

