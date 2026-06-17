# CASTOR

Applies the [DeGF](../README.md) framework to maritime disaster ship images for ONR research.
Runs on the AART Lab `pleiades` SLURM cluster (`head1.condo.cs.cmu.edu`) via Apptainer.

## Quick Start

```bash
# 1. Log into the cluster
ssh <username>@head1.condo.cs.cmu.edu

# 2. Clone / copy the repo into your home directory
#    (images travel with the repo — they fit in the 500 GB home quota)
cd ~
# scp or git clone into ~/DeGF/

# 3. Baseline run (no diffusion)
cd ~/DeGF
sbatch CASTOR/submit_job.sh

# 4. DeGF run (SD reference image + corrective decoding)
sbatch CASTOR/submit_job.sh --use-diffusion
```

The first submission builds the Apptainer container (~10-20 min). Subsequent runs skip the build because `submit_job.sh` hashes `container.def` and only rebuilds when the hash changes.

If you update `container.def` and need a fresh container:
```bash
rm /data/$USER/castor.sif
sbatch CASTOR/submit_job.sh
```

## Commands

| Command | Description |
|---------|-------------|
| `sbatch CASTOR/submit_job.sh` | Submit baseline run |
| `sbatch CASTOR/submit_job.sh --use-diffusion` | Submit DeGF run |
| `squeue -u $USER` | Check job status |
| `tail -f logs/castor_<JOBID>.out` | Stream live log |
| `sacct -j <JOBID> --format=JobID,State,Elapsed` | Check elapsed time after completion |
| `python CASTOR/prepare_dataset.py --image-dir ... --output ...` | Rebuild questions.jsonl from images |

### Interactive debug session

```bash
srun -p pleiades --time=1:00:00 --cpus-per-task=4 --gpus=1 --mem=40G --pty bash
cd ~/DeGF
# Run a single image manually inside the container:
apptainer exec --nv \
    --bind /data/$USER:/data/$USER \
    --bind $HOME/DeGF:$HOME/DeGF \
    /data/$USER/castor.sif \
    python CASTOR/run_inference.py --max-new-tokens 64
```

## Storage Layout

| What | Where | Why |
|------|-------|-----|
| Code + images | `~/DeGF/` (home, 500 GB) | Fast access, backed up, fits comfortably |
| LLaVA-1.5-7B weights | `/data/$USER/llava-v1.5-7b/` | Large model needs 1.9 TB quota |
| Apptainer container | `/data/$USER/castor.sif` | ~6 GB, doesn't fit well in home |
| HF cache (SD model etc.) | `/data/$USER/.cache/huggingface/` | Downloaded once, reused across runs |
| Results | `/data/$USER/castor_results/` | Output can be large across many runs |

**Never write to `/data/shared/`** — students have read-only access.

LLaVA weights are auto-downloaded on first run if `/data/$USER/llava-v1.5-7b/` is empty.

## Output Format

Results land in `/data/$USER/castor_results/`:

| Mode | File |
|------|------|
| Baseline | `answers_baseline.jsonl` |
| DeGF | `answers_degf.jsonl` |

Each line is a JSON record:
```json
{
  "question_id": 0,
  "image": "category/filename.jpg",
  "prompt": "Describe this vessel...",
  "text": "<model answer>",
  "model_id": "llava-v1.5-7b",
  "use_diffusion": false,
  "timing": {"desc_s": 0, "sd_s": 0, "infer_s": 4.2, "total_s": 4.3}
}
```

OOM-skipped images write an error record instead of an answer:
```json
{"question_id": 7, "image": "...", "error": "oom-skip"}
```
This keeps the file's line count aligned with questions-attempted so resume works correctly.

Runs are **resumable**: re-submitting the same job skips already-written lines.

## Overriding Config

All `config.json` values can be overridden from the command line:

```bash
# Tune DeGF weights
sbatch CASTOR/submit_job.sh --use-diffusion --degf-alpha-pos 5.0 --degf-alpha-neg 0.5

# Write to a custom path (disables auto-suffix)
sbatch CASTOR/submit_job.sh --answers-file /data/$USER/castor_results/exp1.jsonl

# Point at a different image set
sbatch CASTOR/submit_job.sh --image-folder CASTOR/shipwreck_wiki_images/subset
```

Run `python CASTOR/run_inference.py --help` for the full list.

## Architecture

```
submit_job.sh
  └─ apptainer exec castor.sif
       └─ run_inference.py
            ├─ load LLaVA-1.5-7B  (experiments/llava/)
            ├─ [DeGF only] load SD v1.5  (degf_utils/image_generation.py)
            └─ per image:
                 ├─ [DeGF only] description pass → SD → reference image
                 ├─ main inference pass  (degf_utils/degf_sample.py monkey-patches
                 │   transformers.GenerationMixin.sample to inject JS-divergence
                 │   based complementary/contrastive decoding)
                 └─ write to answers_*.jsonl
```

Key non-obvious design points:
- `experiments/llava/` is the LLaVA source library (vendored, not a pip package). `run_inference.py` adds `experiments/` to `sys.path` at startup.
- `degf_utils/degf_sample.py` monkey-patches `GenerationMixin.sample` and `greedy_search` at import time (`evolve_degf_sampling()` is called once at module load).
- The Apptainer container is rebuilt only when `container.def` changes (sha256 hash comparison), not on every job.
- `$USER` in `config.json` paths is expanded at runtime via `os.path.expandvars()` — no manual username editing needed.

## Known Gotchas

- **Re-running a job resumes from where it left off.** To start fresh, delete or rename the `.jsonl` output file.
- **SD model downloads on first DeGF run** (~4 GB to `/data/$USER/.cache/huggingface/`). The download is one-time and subsequent runs reuse the cache.
- **OOM errors produce skip records, not missing lines.** Filter `error` key in post-processing.
- **DeGF mode is ~3× slower** than baseline due to the description pass + SD generation.

## Related Docs

- [ADR-001: Apptainer over conda](docs/decisions/ADR-001-apptainer.md)
- [ADR-002: Storage layout on pleiades](docs/decisions/ADR-002-storage-layout.md)
