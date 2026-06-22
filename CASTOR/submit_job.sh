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
#   srun -p pleiades --time=1:00:00 --cpus-per-task=4 --gpus=1 --mem=40G --pty bash
#   cd ~/DeGF
#   apptainer exec --containall --nv \
#       --bind /data/$USER:/data/$USER --bind ~/DeGF:~/DeGF --bind /tmp:/tmp \
#       /data/$USER/castor.sif /opt/conda/bin/python3 CASTOR/run_inference.py
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH -J castor
#SBATCH --exclude=pleiades-1-3
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

# ── Run inference ─────────────────────────────────────────────────────────────
time $APPTAINER_BASE "$SIF" $PYTHON "$REPO/CASTOR/run_inference.py" "$@"

ELAPSED=$(( SECONDS - JOB_START ))
echo "=========================================="
echo " Finished     : $(date)"
echo " Job wall time: $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m $(( ELAPSED%60 ))s"
echo "=========================================="
