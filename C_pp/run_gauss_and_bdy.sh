#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults (overridable)
M=${M:-200}
R=${R:-2}
SIGN=${SIGN:-"+"}
TIMES=${TIMES:-"200"}
SIZES=${SIZES:-"500"}
GAMMAS=${GAMMAS:-"0.1 0.5 1.0"}
GAUSS_ORDER=${GAUSS_ORDER:-128}
MINREC=${MINREC:-4}
MAXSPLITS=${MAXSPLITS:-40} # 0 means no cap (use 2^MINREC)

echo "Compiling main_gauss..."
g++ -std=c++17 -O2 main_gauss.cpp -I/opt/homebrew/include/eigen3 -o main_gauss

for g in $GAMMAS; do
  echo "Running main_gauss gamma=$g M=$M r=$R sign=$SIGN order=$GAUSS_ORDER minrec=$MINREC maxsplits=$MAXSPLITS"
  ./main_gauss "$g" "$M" "$R" "$SIGN" "$GAUSS_ORDER" "$MINREC" "$MAXSPLITS"
done

echo "Running main_dynamics_BDY.py with r=$R sign=$SIGN"
python3 main_dynamics_BDY.py \
  --r "$R" \
  --sign "$SIGN" \
  --sizes $SIZES \
  --times $TIMES \
  --gammas $GAMMAS \
  --qags-sign "$SIGN" \
  --qags-M "$M" \
  --qags-dir . \
  --qags-pattern "GHD_r{r}_sign{sign}_M{M}_gamma{gamma:.6f}_gauss.csv"

