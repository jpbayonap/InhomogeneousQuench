#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"
START_EPOCH=$(date +%s)
echo "run_cov_local.sh started at $(date -Iseconds)"

# Keep BLAS single-threaded; for large covariance jobs parallelize outside Python.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export GOTO_NUM_THREADS=${GOTO_NUM_THREADS:-1}

VENV_PATH=${VENV_PATH:-""}
if [ -n "${VENV_PATH}" ] && [ -d "${VENV_PATH}" ]; then
  # shellcheck disable=SC1090
  source "${VENV_PATH}/bin/activate"
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
N_JOBS=${N_JOBS:-2}

# Defaults for the local corrected covariance run.
# Note: SIZES is the half-chain size L, so SIZES=1000 means total N=2000.
SIZES=${SIZES:-"1000"}
TIMES=${TIMES:-"100 150 200 250 300 400"}
GAMMAS=${GAMMAS:-"0.0"}

# Leave S_OFFSET unset by default so main_dynamics_it_cov.py uses s_offset=size.
S_OFFSET=${S_OFFSET:-""}

BETA=${BETA:-0.0}
BETA_L=${BETA_L:-""}
BETA_R=${BETA_R:-""}
INIT_STATE=${INIT_STATE:-""}
VAC_NEEL=${VAC_NEEL:-0}
NEEL_EVEN=${NEEL_EVEN:-0}
VAC_FILL=${VAC_FILL:-0}
VAC_INFTY=${VAC_INFTY:-0}
PHSYMM=${PHSYMM:-0}
PHSYMM_ODD=${PHSYMM_ODD:-0}
PHSYMM_M=${PHSYMM_M:-1}
PHSYMM_A=${PHSYMM_A:-0.2}

METHOD=${METHOD:-rk4}
RK_STEPS=${RK_STEPS:-600}
RK_DT_MAX=${RK_DT_MAX:-0.40}
RK_TOL=${RK_TOL:-1e-7}
RK_ADAPT=${RK_ADAPT:-0}

COV_DTYPE=${COV_DTYPE:-complex64}
SKIP_EXISTING=${SKIP_EXISTING:-1}
CHECK_PROJ_ROW=${CHECK_PROJ_ROW:-0}
OUTDIR=${OUTDIR:-"$(pwd)/runs/neel_cov_local"}

read -r -a SIZES_ARR <<< "${SIZES}"
read -r -a TIMES_ARR <<< "${TIMES}"
read -r -a GAMMAS_ARR <<< "${GAMMAS}"

echo "Using PYTHON_BIN=$PYTHON_BIN, N_JOBS=$N_JOBS, OUTDIR=$OUTDIR"
echo "SIZES=$SIZES | TIMES=$TIMES | GAMMAS=$GAMMAS"
if [ -n "${S_OFFSET}" ]; then
  echo "S_OFFSET=$S_OFFSET"
else
  echo "S_OFFSET=<default=size>"
fi
echo "BETA=${BETA}, BETA_L=${BETA_L:-<none>}, BETA_R=${BETA_R:-<none>}"
echo "INIT_STATE=${INIT_STATE:-<legacy-infer>} | VAC_NEEL=${VAC_NEEL} | NEEL_EVEN=${NEEL_EVEN} | VAC_FILL=${VAC_FILL} | VAC_INFTY=${VAC_INFTY} | PHSYMM=${PHSYMM} | PHSYMM_ODD=${PHSYMM_ODD} (m=${PHSYMM_M}, A=${PHSYMM_A})"
echo "METHOD=${METHOD} | RK_STEPS=${RK_STEPS} | RK_DT_MAX=${RK_DT_MAX} | RK_TOL=${RK_TOL} | RK_ADAPT=${RK_ADAPT}"
echo "COV_DTYPE=${COV_DTYPE} | SKIP_EXISTING=${SKIP_EXISTING} | CHECK_PROJ_ROW=${CHECK_PROJ_ROW}"

if [ -z "${INIT_STATE}" ]; then
  if [ -n "${BETA_L}" ] || [ -n "${BETA_R}" ]; then
    INIT_STATE=beta_lr
  elif [ "${VAC_NEEL}" != "0" ]; then
    INIT_STATE=neel
  elif [ "${NEEL_EVEN}" != "0" ]; then
    INIT_STATE=neel_even
  elif [ "${VAC_FILL}" != "0" ]; then
    INIT_STATE=vac_fill
  elif [ "${VAC_INFTY}" != "0" ]; then
    INIT_STATE=vac_infty
  elif [ "${PHSYMM}" != "0" ]; then
    INIT_STATE=phsymm
  elif [ "${PHSYMM_ODD}" != "0" ]; then
    INIT_STATE=phsymm_odd
  elif [ "${BETA}" = "0" ] || [ "${BETA}" = "0.0" ]; then
    INIT_STATE=neel
  else
    INIT_STATE=beta
  fi
fi

CMD=(
  "${PYTHON_BIN}" ./main_dynamics_it_cov.py
  --sizes "${SIZES_ARR[@]}"
  --times "${TIMES_ARR[@]}"
  --gammas "${GAMMAS_ARR[@]}"
  --init-state "${INIT_STATE}"
  --method "${METHOD}"
  --rk-steps "${RK_STEPS}"
  --rk-dt-max "${RK_DT_MAX}"
  --rk-tol "${RK_TOL}"
  --n-jobs "${N_JOBS}"
  --cov-dtype "${COV_DTYPE}"
  --outdir "${OUTDIR}"
)

if [ -n "${S_OFFSET}" ]; then
  read -r -a SOFF_ARR <<< "${S_OFFSET}"
  CMD+=(--s-offset "${SOFF_ARR[@]}")
fi
if [ -n "${BETA_L}" ]; then CMD+=(--betaL "${BETA_L}"); fi
if [ -n "${BETA_R}" ]; then CMD+=(--betaR "${BETA_R}"); fi
CMD+=(--beta "${BETA}")
if [ "${INIT_STATE}" = "phsymm" ] || [ "${INIT_STATE}" = "phsymm_odd" ]; then
  CMD+=(--phsymm-m "${PHSYMM_M}" --phsymm-A "${PHSYMM_A}")
fi
if [ "${RK_ADAPT}" != "0" ]; then CMD+=(--rk-adapt); fi
if [ "${SKIP_EXISTING}" != "0" ]; then CMD+=(--skip-existing); fi
if [ "${CHECK_PROJ_ROW}" != "0" ]; then CMD+=(--check-proj-row); fi

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

time "${CMD[@]}"

END_EPOCH=$(date +%s)
ELAPSED=$((END_EPOCH - START_EPOCH))
echo "run_cov_local.sh finished at $(date -Iseconds) | elapsed=${ELAPSED}s"
