#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

# Defaults (can be overridden via env vars)
TIMES=${TIMES:-"200"}
M=${M:-200}
R=${R:-2}
SIGN=${SIGN:-"+"}
GAMMAS=${GAMMAS:-"0.1 0.5 1.0"}
SIZES=${SIZES:-"500"}
EPSABS=${EPSABS:-1e-15}
EPSREL=${EPSREL:-1e-15}
LIMIT=${LIMIT:-100000}
MINREC=${MINREC:-40}
MAXREC=${MAXREC:-100000}

echo "Compiling main_qags..."
g++ -std=c++17 -O2 main_qags.cpp -I/opt/homebrew/include/eigen3 -I/opt/homebrew/include -L/opt/homebrew/lib -lgsl -lgslcblas -o main_qags

for g in $GAMMAS; do
  echo "Running main_qags gamma=$g M=$M r=$R sign=$SIGN epsabs=$EPSABS epsrel=$EPSREL limit=$LIMIT minrec=$MINREC maxrec=$MAXREC"
  ./main_qags "$g" "$M" "$R" "$SIGN" "$EPSABS" "$EPSREL" "$LIMIT" "$MINREC" "$MAXREC"
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
  --qags-dir .
