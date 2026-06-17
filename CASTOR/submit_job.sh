#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CASTOR — SLURM batch job for pleiades (AART Lab, head1.condo.cs.cmu.edu)
#
# Submit from ~/DeGF/:
#   sbatch CASTOR/submit_job.sh                   # baseline
#   sbatch CASTOR/submit_job.sh --use-diffusion   # DeGF run
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/castor_<JOBID>.out
#
# Interactive debug:
#   srun -p pleiades --time=1:00:00 --cpus-per-task=4 --gpus=1 --mem=40G --pty bash
#   cd ~/DeGF
#   apptainer build --fakeroot /data/$USER/castor.sif CASTOR/container.def
#   apptainer exec --nv --bind /data/$USER:/data/$USER /data/$USER/castor.sif \
#       python CASTOR/run_inference.py --use-diffusion
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH -o logs/castor_%j.out
#SBATCH -e logs/castor_%j.err
#SBATCH -J castor

set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
mkdir -p logs

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

# ── Download LLaVA-1.5-7B if not already present ─────────────────────────────
MODEL_DIR="$DATA_DIR/llava-v1.5-7b"
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "[$(date)] Downloading LLaVA-1.5-7B → $MODEL_DIR ..."
    apptainer exec \
        --bind "$DATA_DIR:$DATA_DIR" \
        "$SIF" \
        python -c "
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
time apptainer exec \
    --nv \
    --bind "$REPO:$REPO" \
    --bind "$DATA_DIR:$DATA_DIR" \
    "$SIF" \
    python "$REPO/CASTOR/run_inference.py" "$@"

ELAPSED=$(( SECONDS - JOB_START ))
echo "=========================================="
echo " Finished     : $(date)"
echo " Job wall time: $(( ELAPSED/3600 ))h $(( (ELAPSED%3600)/60 ))m $(( ELAPSED%60 ))s"
echo "=========================================="
