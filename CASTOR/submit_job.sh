#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CASTOR — SLURM batch job for pleiades (AART Lab, head1.condo.cs.cmu.edu)
#
# Do NOT call this file directly with sbatch — use CASTOR/submit.sh instead.
# submit.sh creates /data/$USER/logs/ before sbatch opens the log file.
#
# Submit from ~/DeGF/:
#   bash CASTOR/submit.sh                                          # → answers_baseline.jsonl
#   bash CASTOR/submit.sh --use-diffusion                         # → answers_degf.jsonl
#   bash CASTOR/submit.sh --use-diffusion --run-name ap5_b02      # → answers_degf_ap5_b02.jsonl
#
# Monitor:
#   squeue -u $USER
#   tail -f /data/$USER/logs/castor_<JOBID>.out
#
# Interactive debug:
#   srun -p pleiades --time=1:00:00 --cpus-per-task=4 --gpus=1 --mem=40G --constraint=RTX6000ADA --pty bash
#   cd ~/DeGF
#   apptainer exec --containall --nv \
#       --bind /data/$USER:/data/$USER --bind ~/DeGF:~/DeGF --bind /tmp:/tmp \
#       /data/$USER/castor.sif /opt/conda/bin/python3 CASTOR/run_inference.py
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH -J castor
#SBATCH --constraint=RTX6000ADA
# NOTE: --output and --error are set by CASTOR/submit.sh to /data/$USER/logs/

set -e
# SLURM_SUBMIT_DIR is the directory where sbatch was called (~/DeGF/).
# Avoids relying on $0 path resolution which can vary across SLURM versions.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
mkdir -p "/data/$USER/logs"

JOB_START=$SECONDS

echo "=========================================="
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Started  : $(date)"
echo " Args     : $@"
echo " User     : $USER"
echo " Repo     : $REPO"
echo "=========================================="

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ── All user-writable paths live under /data/$USER/ ──────────────────────────
DATA_DIR="/data/$USER"
export HF_HOME="$DATA_DIR/.cache/huggingface"
export TRANSFORMERS_CACHE="$DATA_DIR/.cache/huggingface"
export TORCH_HOME="$DATA_DIR/.cache/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# ── Build Apptainer container if missing or container.def has changed ─────────
SIF="$DATA_DIR/castor.sif"
DEF_HASH=$(sha256sum CASTOR/container.def | cut -d' ' -f1)
SIF_HASH_FILE="$SIF.def.sha256"
if [ ! -f "$SIF" ] || [ ! -f "$SIF_HASH_FILE" ] || [ "$DEF_HASH" != "$(cat "$SIF_HASH_FILE")" ]; then
    echo "[$(date)] Building container from CASTOR/container.def (hash: $DEF_HASH) ..."
    if apptainer build --fakeroot "$SIF" CASTOR/container.def; then
        echo "$DEF_HASH" > "$SIF_HASH_FILE"
        echo "[$(date)] Container ready: $SIF"
    else
        echo "[$(date)] Container build FAILED — check logs above." >&2
        exit 1
    fi
else
    echo "[$(date)] Container up-to-date (hash: $DEF_HASH), skipping build."
fi

# --containall stops the cluster's apptainer.conf from bind-mounting the host
# /opt over the container's /opt/conda. We then add back only what's needed.
# --env passes shell variables that --containall/--cleanenv would otherwise strip.
APPTAINER_BASE="apptainer exec --containall --nv \
    --pwd $REPO \
    --env USER=$USER \
    --env HOME=$HOME \
    --env HF_HOME=$HF_HOME \
    --env TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE \
    --env TORCH_HOME=$TORCH_HOME \
    --bind /tmp:/tmp \
    --bind $REPO:$REPO \
    --bind $DATA_DIR:$DATA_DIR"
PYTHON=/opt/conda/bin/python3

echo "[$(date)] Container Python: $PYTHON"

# ── Download LLaVA-1.5-7B if not already present ─────────────────────────────
MODEL_DIR="$DATA_DIR/llava-v1.5-7b"
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "[$(date)] Downloading LLaVA-1.5-7B → $MODEL_DIR ..."
    $APPTAINER_BASE "$SIF" $PYTHON -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'liuhaotian/llava-v1.5-7b',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False,
)
print('LLaVA download complete.')
"
    echo "[$(date)] LLaVA ready at $MODEL_DIR"
else
    echo "[$(date)] LLaVA already present at $MODEL_DIR — skipping download"
fi

# ── Select prompt and mode for this array task ───────────────────────────────
# submit.sh uses interleaved task IDs:
#   both modes (default) → even task ID = baseline, odd = degf (same prompt)
#   one mode specified   → task ID maps directly to prompt index
PROMPTS_DIR="$REPO/CASTOR/prompts"
IMAGE_DIR="$REPO/CASTOR/shipwreck_wiki_images/sorted_images"

PROMPT_FILES=( "$PROMPTS_DIR"/*.txt )
N_PROMPTS=${#PROMPT_FILES[@]}

# ── Parse "$@" in one pass ────────────────────────────────────────────────────
USER_RUN_NAME=""
HAS_USE_DIFFUSION=false
HAS_NO_DIFFUSION=false
PASSTHROUGH=()
_args=("$@"); _i=0
while [[ $_i -lt ${#_args[@]} ]]; do
    case "${_args[$_i]}" in
        --run-name)      _i=$((_i+1)); USER_RUN_NAME="${_args[$_i]}" ;;
        --run-name=*)    USER_RUN_NAME="${_args[$_i]#--run-name=}" ;;
        --use-diffusion) HAS_USE_DIFFUSION=true ;;
        --no-diffusion)  HAS_NO_DIFFUSION=true  ;;
        *)               PASSTHROUGH+=("${_args[$_i]}") ;;
    esac
    _i=$((_i+1))
done
unset _args _i

# ── Resolve prompt file and mode from task ID ─────────────────────────────────
if $HAS_USE_DIFFUSION; then
    PROMPT_IDX=$SLURM_ARRAY_TASK_ID
    MODE_FLAG="--use-diffusion"
elif $HAS_NO_DIFFUSION; then
    PROMPT_IDX=$SLURM_ARRAY_TASK_ID
    MODE_FLAG="--no-diffusion"
else
    # Both modes: even task → baseline, odd task → degf (pairs share a prompt)
    PROMPT_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
    if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then
        MODE_FLAG="--no-diffusion"
    else
        MODE_FLAG="--use-diffusion"
    fi
fi

PROMPT_FILE="${PROMPT_FILES[$PROMPT_IDX]}"
STEM=$(basename "$PROMPT_FILE" .txt)

# ── Build run name ────────────────────────────────────────────────────────────
# Pattern: [user_tag_]{stem}_j{ArrayJobID}
# Matches log file: castor[_{user_tag}]_{ArrayJobID}_{TaskID}.out
if [[ -n "$USER_RUN_NAME" ]]; then
    RUN_NAME="${USER_RUN_NAME}_${STEM}_j${SLURM_ARRAY_JOB_ID}"
else
    RUN_NAME="${STEM}_j${SLURM_ARRAY_JOB_ID}"
fi

QUESTIONS_FILE="$DATA_DIR/castor_results/questions_${RUN_NAME}.jsonl"
mkdir -p "$DATA_DIR/castor_results"

echo "=========================================="
echo " Array task  : $SLURM_ARRAY_TASK_ID  (job $SLURM_ARRAY_JOB_ID)"
echo " Mode        : $MODE_FLAG"
echo " Prompt file : $PROMPT_FILE"
echo " Run name    : $RUN_NAME"
echo " Questions   : $QUESTIONS_FILE"
echo "=========================================="

# ── Prepare dataset with this prompt ─────────────────────────────────────────
$APPTAINER_BASE "$SIF" $PYTHON "$REPO/CASTOR/prepare_dataset.py" \
    --image-dir   "$IMAGE_DIR" \
    --output      "$QUESTIONS_FILE" \
    --prompt-file "$PROMPT_FILE"

# ── Run inference ─────────────────────────────────────────────────────────────
time $APPTAINER_BASE "$SIF" $PYTHON "$REPO/CASTOR/run_inference.py" \
    "${PASSTHROUGH[@]}" \
    --question-file "$QUESTIONS_FILE" \
    --run-name      "$RUN_NAME" \
    "$MODE_FLAG"

ELAPSED=$(( SECONDS - JOB_START ))
echo "=========================================="
echo " Finished     : $(date)"
echo " Job wall time: $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m $(( ELAPSED%60 ))s"
echo "=========================================="
