#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --constraint=matrix
#SBATCH --partition=m3
#SBATCH --job-name="ghd_hybrid_m3"
#SBATCH --output=slurm-ghd-hybrid-%j.out
#SBATCH --error=slurm-ghd-hybrid-%j.err
#SBATCH --mail-user=juanpablo.bayonapen2@unibo.it
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32G
#SBATCH --qos=normal
#SBATCH --time=23:59:59
#SBATCH --chdir=/home/PERSONALE/juanpablo.bayonapen2/InhomogeneousQuench/C_pp/mathematica_scripts

set -euo pipefail

REPO_DIR="/home/PERSONALE/juanpablo.bayonapen2/InhomogeneousQuench"
SCRIPT_DIR="$REPO_DIR/C_pp/mathematica_scripts"

# Path to the sandbox container on the cluster account.
SANDBOX_PATH="${SANDBOX_PATH:-/home/PERSONALE/juanpablo.bayonapen2/test/ubuntu16}"

# Bind the whole repo so the Mathematica script can read/write under C_pp.
HOST_DIR="$REPO_DIR"
CONTAINER_DIR="/mnt/repo"

USERNAME=$(whoami)
DATETIME=$(date +"%Y%m%d_%H%M%S")
INSTANCE_NAME="${USERNAME}_${DATETIME}"

# Use a cleaned batch .wl script by default.
MMA_SCRIPT="${MMA_SCRIPT:-GHD_hybrid_batch.wl}"
MMA_NUM_KERNELS="${MMA_NUM_KERNELS:-$SLURM_CPUS_PER_TASK}"
MMA_EXEC="/mnt/ffhiggstop/Wolfram/Mathematica/13.3/Executables/math"
TARGET_DIR="$CONTAINER_DIR/C_pp/mathematica_scripts"

if [[ ! -d "$SANDBOX_PATH" ]]; then
  echo "Sandbox container not found: $SANDBOX_PATH" >&2
  exit 1
fi

if [[ ! -f "$SCRIPT_DIR/$MMA_SCRIPT" ]]; then
  echo "Mathematica script not found: $SCRIPT_DIR/$MMA_SCRIPT" >&2
  echo "Set MMA_SCRIPT to the cleaned batch .wl file before submitting." >&2
  exit 1
fi

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "MMA_NUM_KERNELS=$MMA_NUM_KERNELS"
echo "SANDBOX_PATH=$SANDBOX_PATH"
echo "HOST_DIR=$HOST_DIR"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "MMA_SCRIPT=$MMA_SCRIPT"
echo "INSTANCE_NAME=$INSTANCE_NAME"

cleanup() {
  apptainer instance stop "$INSTANCE_NAME" >/dev/null 2>&1 || true
}

trap cleanup EXIT

apptainer instance start -w --bind "$HOST_DIR:$CONTAINER_DIR" "$SANDBOX_PATH" "$INSTANCE_NAME"

apptainer exec "instance://$INSTANCE_NAME" bash -lc "
  set -euo pipefail
  echo 'Adding /mnt/ffhiggstop/Wolfram/Mathematica/13.3/Executables to PATH ...'
  export PATH=\$PATH:/mnt/ffhiggstop/Wolfram/Mathematica/13.3/Executables
  export MMA_NUM_KERNELS=$MMA_NUM_KERNELS

  echo 'Step 1: Printing current directory ...'
  pwd

  echo 'Step 2: Changing to target directory inside container ...'
  cd $TARGET_DIR

  echo 'Step 3: Current directory is now:'
  pwd

  echo 'Step 4: Running Mathematica script: $MMA_SCRIPT ...'
  $MMA_EXEC -script $MMA_SCRIPT

  echo 'Raga, Mathematica script execution completed!'
"
