#!/bin/bash
# Wrapper around sbatch: creates the writable log dir BEFORE sbatch opens the
# log file, counts prompts, and submits one array task per (prompt × mode) pair
# so everything runs in parallel — each task gets its own GPU allocation.
#
# Usage (from ~/BenchyBench/DeGF/):
#   bash CASTOR/submit.sh                    # N prompts × 2 modes = 2N tasks
#   bash CASTOR/submit.sh --use-diffusion    # N tasks, degf only
#   bash CASTOR/submit.sh --no-diffusion     # N tasks, baseline only
#   bash CASTOR/submit.sh --run-name exp1    # tag all output/log files
#
# Log files:   /data/$USER/logs/castor[_{run_name}]_{ArrayJobID}_{TaskID}.out
# Output files:/data/$USER/castor_results/answers_{mode}[_{run_name}]_{stem}_j{ArrayJobID}.jsonl
#
# Monitor:
#   squeue -u $USER
#   tail -f /data/$USER/logs/castor_*<ARRAYJOBID>*.out

LOG_DIR="/data/$USER/logs"
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROMPTS_DIR="$SCRIPT_DIR/prompts"
N=$(ls "$PROMPTS_DIR"/*.txt 2>/dev/null | wc -l)
if [[ "$N" -eq 0 ]]; then
    echo "ERROR: No .txt files found in $PROMPTS_DIR" >&2
    exit 1
fi

# Parse args: detect mode flags and --run-name for log naming.
RUN_NAME_TAG=""
HAS_USE_DIFFUSION=false
HAS_NO_DIFFUSION=false
_args=("$@"); _i=0
while [[ $_i -lt ${#_args[@]} ]]; do
    case "${_args[$_i]}" in
        --run-name)      _i=$((_i+1)); RUN_NAME_TAG="${_args[$_i]}" ;;
        --run-name=*)    RUN_NAME_TAG="${_args[$_i]#--run-name=}" ;;
        --use-diffusion) HAS_USE_DIFFUSION=true ;;
        --no-diffusion)  HAS_NO_DIFFUSION=true  ;;
    esac
    _i=$((_i+1))
done
unset _args _i

# One mode specified → N tasks; both modes (default) → 2N tasks.
# Even task IDs = baseline, odd task IDs = degf (interleaved per prompt).
if $HAS_USE_DIFFUSION || $HAS_NO_DIFFUSION; then
    ARRAY_END=$(( N - 1 ))
    echo "Submitting array job: $N tasks ($N prompts, 1 mode)"
else
    ARRAY_END=$(( N * 2 - 1 ))
    echo "Submitting array job: $(( N * 2 )) tasks ($N prompts × 2 modes)"
fi

# Log prefix matches output file naming: castor[_{run_name}]_{ArrayJobID}_{TaskID}.out
LOG_PREFIX="castor${RUN_NAME_TAG:+_${RUN_NAME_TAG}}"

# ── Container: build once before array tasks start ────────────────────────────
# apptainer build cannot run on the head node; submit a dedicated build job and
# chain the array job behind it with --dependency=afterok so tasks only start
# after a successful build. If the container is already up-to-date, skip.
DATA_DIR="/data/$USER"
SIF="$DATA_DIR/castor.sif"
DEF="$SCRIPT_DIR/container.def"
DEF_HASH=$(sha256sum "$DEF" | cut -d' ' -f1)
SIF_HASH_FILE="$SIF.def.sha256"

DEPENDENCY=""
if [ -f "$SIF" ] && [ -f "$SIF_HASH_FILE" ] && [ "$DEF_HASH" = "$(cat "$SIF_HASH_FILE")" ]; then
    echo "[container] Up-to-date (hash: $DEF_HASH), skipping build."
else
    echo "[container] Stale or missing — submitting build job (hash: $DEF_HASH) ..."
    BUILD_JOB=$(sbatch --parsable \
        --output="$LOG_DIR/build_castor_%j.out" \
        --error="$LOG_DIR/build_castor_%j.err" \
        "$SCRIPT_DIR/build_container.sh")
    echo "[container] Build job $BUILD_JOB submitted — array will wait for it."
    DEPENDENCY="--dependency=afterok:${BUILD_JOB}"
fi

exec sbatch \
    $DEPENDENCY \
    --output="$LOG_DIR/${LOG_PREFIX}_%A_%a.out" \
    --error="$LOG_DIR/${LOG_PREFIX}_%A_%a.err" \
    --array="0-${ARRAY_END}" \
    "$SCRIPT_DIR/submit_job.sh" "$@"
