# ADR-002: Storage layout on the pleiades cluster

## Status
Accepted

## Date
2026-06-17

## Context
pleiades has three relevant storage areas with different quotas and permissions:

| Path | Quota | Access |
|------|-------|--------|
| `/home/<user>/` | 500 GB | Read/write (personal) |
| `/data/<user>/` | 1.9 TB | Read/write (personal) |
| `/data/shared/` | shared | Read-only for students |

The CASTOR stack needs to place: code, shipwreck images, LLaVA weights (~14 GB),
Stable Diffusion weights (~4 GB), the Apptainer container (~6 GB), HF cache, and results.

## Decision

| Artifact | Location | Reason |
|----------|----------|--------|
| Code + images | `~/DeGF/` (home) | Comfortably under 500 GB; convenient for editing |
| LLaVA weights | `/data/$USER/llava-v1.5-7b/` | 14 GB too large for the home quota over time |
| Apptainer `.sif` | `/data/$USER/castor.sif` | ~6 GB; home quota would be tight |
| HF model cache | `/data/$USER/.cache/huggingface/` | SD model (~4 GB) + other downloads |
| Results | `/data/$USER/castor_results/` | Can grow large across experiments |

`$USER` is expanded at runtime by `os.path.expandvars()` in `run_inference.py` — no
hardcoded username in any config or script.

## Alternatives Considered

### Everything in `/data/shared/`
- Rejected: students have read-only access; cannot write results or cache

### Everything in home
- Rejected: LLaVA + SD + container + results would approach the 500 GB limit quickly

### Everything in `/data/$USER/`
- Pros: Avoids home quota pressure entirely
- Cons: Cluster home directories are backed up; `/data` typically is not. Keeping
  code in home provides an implicit backup.
- Rejected: Code is under version control (git) so backup is less critical, but
  keeping code in home follows AART Lab convention and allows easier `scp`/editing

## Consequences
- `submit_job.sh` auto-creates `$DATA_DIR/.cache/huggingface` and `.cache/torch`
  before the container starts, so the first run doesn't fail on missing directories.
- LLaVA weights are auto-downloaded on first run if the target directory is missing
  or empty. The download runs inside the container (same dependency set) without GPU.
- `/data/$USER/` is bound into the container via `--bind $DATA_DIR:$DATA_DIR` so
  the same absolute path resolves inside and outside the container.
