#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

SIGN=${SIGN:-"+"}
R_LIST =${R_LIST:-"1"}
GAMMAS=${GAMMAS:-" 0.5 1.0 2.0 4.0"}
SIZES=${SIZES:-8}
TIMES=${TIMES:-"250"}
BETA_L=${BETA_L:-1.0}
BETA_R=${BETA_R:-1.0}
L=${L:-"300"}

for R in $R_LIST; do

    python3 plot_neel_superposed.py\
    --r "$R"\
    --sign "$SIGN"  \
    --gamma $GAMMAS \
    --time $TIMES   \
    --N    $SIZES   \
