#!/bin/bash

#SBATCH --output=%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --cpus-per-task=16
#SBATCH --exclude=artongpu01,tikgpu[01-10]
#SBATCH --array=0-1

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Change the current directory to the location where you want to store temporary files, exit if changing didn't succeed.
# Adapt this to your personal preference
cd "${TMPDIR}" || exit 1

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Set WandB dir
# export WANDB_CONFIG_DIR=/itet-stor/mlundwall/net_scratch/wandb

# Binary or script to execute
/itet-stor/mlundwall/net_scratch/conda_envs/bees/bin/python /home/mlundwall/bees/train.py $SLURM_ARRAY_TASK_ID

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0
