#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults (override via env)
R=${R:-1}
SIGN=${SIGN:-"-"}
GAMMAS=${GAMMAS:-"0.1"}
SIZES=${SIZES:-"300"}
TIMES=${TIMES:-"100"}
NJOBS=${NJOBS:-1}
OUTDIR=${OUTDIR:-$(pwd)}

python3 Liouvillian_gap.py \
  --r "$R" \
  --sign "$SIGN" \
  --sizes $SIZES \
  --times $TIMES \
  --gammas $GAMMAS \
  --n-jobs "$NJOBS" \
  --outdir "$OUTDIR"
