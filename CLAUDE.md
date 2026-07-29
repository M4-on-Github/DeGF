# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

**DeGF** (ICLR 2025) is a hallucination-mitigation framework for vision-language models that uses Stable Diffusion to generate a "reference image" from the model's own description, then applies JS-divergence-based contrastive decoding.

**CASTOR** (`CASTOR/`) is an ONR research overlay that applies DeGF to maritime disaster ship images, running on the AART Lab `pleiades` SLURM cluster (`head1.condo.cs.cmu.edu`) via Apptainer.

## Hard constraints

- **Never modify `degf_utils/`** — this is the core DeGF algorithm (monkey-patched generation). Any change risks breaking the JS-divergence decoding logic.
- **Never upgrade pinned packages** — `transformers==4.31.0`, `tokenizers==0.13.3`, `peft==0.4.0`, `torch==2.0.1`, `bitsandbytes==0.41.0`, `diffusers==0.21.4`, `torchvision==0.15.2`. These pins are interdependent and were chosen to avoid known breakages. See `CASTOR/container.def` for install-order rationale.
- **Never add Python dependencies** without checking against the container's existing pins.

## Branch layout

| Branch | Purpose |
|--------|---------|
| `interactive_session` | SLURM job scripts, `run_inference.py`, container |
| `castor_evals` | `Eval_CASTOR/` evaluation scripts only |

Keep these branches separate. SLURM/inference work goes on `interactive_session`; eval changes go on `castor_evals`.

## Cluster commands

```bash
# Submit baseline + DeGF in parallel (default: 2N array tasks for N prompts)
bash CASTOR/submit.sh

# Submit one mode only
bash CASTOR/submit.sh --no-diffusion
bash CASTOR/submit.sh --use-diffusion

# Tag output files for an experiment
bash CASTOR/submit.sh --run-name exp1

# Monitor
squeue -u $USER
tail -f /data/$USER/logs/castor_<ArrayJobID>_<TaskID>.out

# Interactive debug (RTX6000Ada nodes only)
srun -p pleiades --time=1:00:00 --cpus-per-task=4 --gpus=1 --mem=40G --constraint=RTX6000ADA --pty bash

# Inside interactive session — run one image directly
cd ~/DeGF
apptainer exec --containall --nv \
    --bind /data/$USER:/data/$USER --bind ~/DeGF:~/DeGF --bind /tmp:/tmp \
    /data/$USER/castor.sif /opt/conda/bin/python3 CASTOR/run_inference.py --no-diffusion

# Rebuild the container (only needed when container.def changes)
rm /data/$USER/castor.sif && bash CASTOR/submit.sh
```

## Inference pipeline architecture

```
CASTOR/submit.sh                  (wrapper: creates log dir, counts prompts, calls sbatch)
  └─ sbatch CASTOR/submit_job.sh  (SLURM array job: one task per prompt × mode pair)
       ├─ builds/reuses castor.sif (hash of container.def, skipped if unchanged)
       ├─ runs prepare_dataset.py  (builds per-prompt questions.jsonl)
       └─ apptainer exec castor.sif → run_inference.py
            ├─ loads LLaVA-1.5-7B  (experiments/llava/ — vendored, not pip)
            ├─ [DeGF only] loads SD v1.5  (degf_utils/image_generation.py)
            └─ per image:
                 ├─ [DeGF only] description pass → SD → reference image
                 ├─ main inference  (degf_utils/degf_sample.py monkey-patches
                 │   GenerationMixin.sample to inject JS-divergence decoding)
                 └─ write to answers_*.jsonl  (resumable: skips already-written lines)
```

### Non-obvious design points

- `experiments/llava/` is the vendored LLaVA source library. `run_inference.py` inserts both `experiments/` and the repo root into `sys.path` at startup — there is no pip-installed `llava` package.
- `degf_utils/degf_sample.py` monkey-patches `transformers.GenerationMixin.sample` **at import time** via `evolve_degf_sampling()`. This call happens before `run_inference.py` loads the model.
- `$USER` in `CASTOR/config.json` paths is expanded at runtime via `os.path.expandvars()` — never hardcode a username.
- All CLI flags to `run_inference.py` override `config.json`. `--answers-file` disables the auto-suffix logic; without it the output is named `answers_{mode}[_{run_name}]_{stem}_j{ArrayJobID}.jsonl`.
- Runs are **resumable**: `run_inference.py` counts existing lines in the output `.jsonl` and skips that many questions. OOM-skipped images write a `{"error": "oom-skip"}` placeholder to keep line counts aligned with question indices.

## SLURM array job task ID mapping

With N prompts and both modes (default):
- Array size: `0` to `2N-1`
- Even task ID → `--no-diffusion` (baseline), `PROMPT_IDX = task_id / 2`
- Odd task ID → `--use-diffusion` (DeGF), `PROMPT_IDX = task_id / 2`

With a single mode flag: array size `0` to `N-1`, task ID maps directly to prompt index.

Log files use `%A` (array job ID) and `%a` (task index); the `j{ArrayJobID}` suffix in output filenames is the linking anchor between logs and results.

## Storage layout on pleiades

| What | Path |
|------|------|
| Code + images | `~/DeGF/` (home, 500 GB quota) |
| LLaVA-1.5-7B weights | `/data/$USER/llava-v1.5-7b/` (auto-downloaded on first run) |
| Apptainer container | `/data/$USER/castor.sif` |
| HF cache (SD weights etc.) | `/data/$USER/.cache/huggingface/` |
| Results + questions files | `/data/$USER/castor_results/` |
| Logs | `/data/$USER/logs/` |

Students have **read-only** access to `/data/shared/` — never write there.

## Prompt sweep

Prompts live in `CASTOR/prompts/*.txt` (one file per field/task, e.g. `1_state.txt`, `2_type.txt`). `prepare_dataset.py` takes a single `--prompt-file` and writes a `questions.jsonl` for that prompt. The SLURM array job calls `prepare_dataset.py` once per task before inference.

To add a new prompt variant: drop a `.txt` file into `CASTOR/prompts/`. The array size is computed automatically from the glob count — no script changes needed.

## Evaluation (castor_evals branch)

`Eval_CASTOR/eval_castor.py` — evaluates combined-field inference output (JSON/CoT extraction via regex + Gemma parser).

`Eval_CASTOR/eval_castor_separated.py` — evaluates separated-prompt runs (one JSONL per field, joined by `image` key). Imports shared helpers (`normalize_state`, `vessel_jaccard`, etc.) directly from `eval_castor.py`.

`Eval_CASTOR/parse_with_gemma.py` — Gemma-based fallback parser for answers that fail regex extraction.

Ground truth: `Eval_CASTOR/human_ground_truth_label/human_gt.csv`.
