#!/bin/bash
# Wrapper around sbatch: creates the writable log dir BEFORE sbatch opens the
# log file, then delegates all arguments to submit_job.sh.
#
# Usage (from ~/DeGF/):
#   bash CASTOR/submit.sh                  # baseline
#   bash CASTOR/submit.sh --use-diffusion  # DeGF run
#
# Monitor:
#   squeue -u $USER
#   tail -f /data/$USER/logs/castor_<JOBID>.out

LOG_DIR="/data/$USER/logs"
mkdir -p "$LOG_DIR"

exec sbatch \
    --output="$LOG_DIR/castor_%j.out" \
    --error="$LOG_DIR/castor_%j.err" \
    "$(dirname "$(realpath "$0")")/submit_job.sh" "$@"
