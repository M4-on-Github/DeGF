#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Build castor.sif from CASTOR/container.def.
# Submitted by CASTOR/submit.sh — do NOT call directly with sbatch.
# The array inference job depends on this job via --dependency=afterok.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -p pleiades
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH -J build_castor
# NOTE: --output and --error are set by CASTOR/submit.sh

set -e
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"

DATA_DIR="/data/$USER"
SIF="$DATA_DIR/castor.sif"
DEF="$REPO/CASTOR/container.def"
DEF_HASH=$(sha256sum "$DEF" | cut -d' ' -f1)

echo "=========================================="
echo " Build job   : $SLURM_JOB_ID"
echo " Node        : $(hostname)"
echo " Started     : $(date)"
echo " Target SIF  : $SIF"
echo " DEF hash    : $DEF_HASH"
echo "=========================================="

mkdir -p "$DATA_DIR"

echo "[$(date)] Running: apptainer build --fakeroot $SIF $DEF"
apptainer build --fakeroot "$SIF" "$DEF"
echo "$DEF_HASH" > "$SIF.def.sha256"
echo "[$(date)] Container ready: $SIF"

# ── Download LLaVA-1.5-7B if not already present ─────────────────────────────
# Done here (single sequential job) to avoid all array tasks racing to download.
MODEL_DIR="$DATA_DIR/llava-v1.5-7b"

APPTAINER_BASE="apptainer exec --containall --nv \
    --pwd $REPO \
    --env USER=$USER \
    --env HOME=$HOME \
    --env HF_HOME=$DATA_DIR/.cache/huggingface \
    --env TRANSFORMERS_CACHE=$DATA_DIR/.cache/huggingface \
    --env TORCH_HOME=$DATA_DIR/.cache/torch \
    --bind /tmp:/tmp \
    --bind $REPO:$REPO \
    --bind $DATA_DIR:$DATA_DIR"
PYTHON=/opt/conda/bin/python3
mkdir -p "$DATA_DIR/.cache/huggingface" "$DATA_DIR/.cache/torch"

if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "[$(date)] Downloading LLaVA-1.5-7B → $MODEL_DIR ..."
    $APPTAINER_BASE "$SIF" $PYTHON -c "
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    'liuhaotian/llava-v1.5-7b',
    local_dir=sys.argv[1],
    local_dir_use_symlinks=False,
)
print('LLaVA download complete.')
" "$MODEL_DIR"
    echo "[$(date)] LLaVA ready at $MODEL_DIR"
else
    echo "[$(date)] LLaVA already present at $MODEL_DIR — skipping download"
fi

echo "=========================================="
echo " Container ready : $SIF"
echo " Model ready     : $MODEL_DIR"
echo " Finished        : $(date)"
echo "=========================================="
