#!/bin/bash
# Wrapper around sbatch: creates the writable log dir BEFORE sbatch opens the
# log file, then delegates all arguments to submit_job.sh.
#
# Usage (from ~/DeGF/):
#   bash CASTOR/submit.sh                                           # → answers_baseline.jsonl
#   bash CASTOR/submit.sh --use-diffusion                          # → answers_degf.jsonl
#   bash CASTOR/submit.sh --use-diffusion --run-name ap5_b02       # → answers_degf_ap5_b02.jsonl
#   bash CASTOR/submit.sh --use-diffusion --degf-alpha-pos 5.0 \
#       --degf-beta 0.2 --run-name ap5_b02                         # named sweep
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
