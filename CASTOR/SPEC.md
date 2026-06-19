# CASTOR Cluster Fix — Spec

Scope: bug fixes only to make `CASTOR/run_inference.py` run reliably on the
pleiades SLURM cluster with GPU (and optionally CPU-only for local testing).
No new features, no refactors outside the affected paths.

---

## 1. Objective

Fix three classes of failure that prevent CASTOR from completing a SLURM job:

| # | Symptom | Root cause |
|---|---------|------------|
| 1 | `mkdir: cannot create directory 'logs': Permission denied` | `#SBATCH -o logs/…` is relative; SLURM opens the file before the script body runs; home-dir is read-only on compute nodes. |
| 2 | Crash or wrong device on server | `image_generation.py` hardcodes `.to("cuda:0")` with no fallback; `run_inference.py` hardcodes `.cuda()` throughout. |
| 3 | Log flood / bitsandbytes noise | Per-token `print()` in `degf_sample.py` emits thousands of lines per image; `BITSANDBYTES_NOWELCOME` unset; `torchvision==0.15.2` not explicitly pinned in container. |

Target users: researcher (sole user) running on AART Lab pleiades cluster via SLURM + Apptainer.

---

## 2. Commands

```bash
# One-time setup on head node (creates writable log dir)
mkdir -p /data/$USER/logs

# Submit baseline run
bash CASTOR/submit.sh

# Submit DeGF run
bash CASTOR/submit.sh --use-diffusion

# Monitor
squeue -u $USER
tail -f /data/$USER/logs/castor_<JOBID>.out

# Interactive debug (allocate node, then run directly)
srun -p pleiades --time=1:00:00 --cpus-per-task=4 --gpus=1 --mem=40G --pty bash
# inside the node:
cd ~/DeGF
apptainer exec --nv \
  --bind /data/$USER:/data/$USER \
  /data/$USER/castor.sif \
  python CASTOR/run_inference.py --no-diffusion
```

---

## 3. Files Changed

| File | Change |
|------|--------|
| `CASTOR/submit.sh` | **NEW** — wrapper that `mkdir`s the log dir then calls `sbatch` with explicit `--output`/`--error` paths |
| `CASTOR/submit_job.sh` | Remove `#SBATCH -o/e` (overridden by wrapper); keep `mkdir -p /data/$USER/logs` as a safety net |
| `CASTOR/container.def` | Pin `torchvision==0.15.2`; add `BITSANDBYTES_NOWELCOME=1` to `%environment` |
| `CASTOR/run_inference.py` | Detect device once (`_DEVICE`); replace hardcoded `.cuda()` / `.half()` with device-aware calls; print GPU name + VRAM at startup (or "CPU" if no GPU found — treated as a fatal warning since the cluster always provides one) |
| `degf_utils/image_generation.py` | Accept optional `device` arg; default to `cuda:0` if CUDA available else `cpu`; use `float32` on CPU |
| `degf_utils/degf_sample.py` | Remove per-token debug `print()` calls (the `use_ritual/use_vcd/…` prints that fire on every token) |

---

## 4. Code Style

- Match existing style: no type annotations added beyond what's already there, no docstrings unless replacing an existing one.
- Each patch is surgical — only the lines that need to change.
- No new abstraction layers; no helper modules; no config keys added.

---

## 5. Testing Strategy

After the fixes, verify in order:

1. **Submit test** — `bash CASTOR/submit.sh` completes without permission errors; `squeue` shows the job queued.
2. **Log file** — `ls /data/$USER/logs/` shows `castor_<JOBID>.out` and `.err`.
3. **Container build** — first run builds the SIF; second run skips build (hash match).
4. **Inference smoke test** — interactive SLURM session, run `python CASTOR/run_inference.py --no-diffusion` on 1 image (subset of questions.jsonl), confirm output in `answers_baseline.jsonl`.
5. **GPU utilization** — `nvidia-smi` inside the job shows memory used (confirms model loaded to GPU, not silently falling back to CPU).

No automated test suite is added (out of scope for a bug fix).

---

## 6. Boundaries

| Category | Rule |
|----------|------|
| **Always do** | GPU is required and must be confirmed at startup — if `torch.cuda.is_available()` is `False`, log a clear error and exit rather than silently running on CPU. |
| **Always do** | Preserve all existing version pins in `container.def` exactly. |
| **Always do** | Keep the Apptainer / SLURM architecture — no change to how the job is structured. |
| **Ask first** | Any change to `config.json` default values. |
| **Ask first** | Any change to `degf_utils/degf_sample.py` logic (JS divergence, decoding weights). |
| **Never** | Upgrade `transformers`, `tokenizers`, `peft`, `torch`, `bitsandbytes`, or `diffusers` versions. |
| **Never** | Add new Python dependencies without checking against the existing pin constraints. |
| **Never** | Modify `shipwreck_wiki_images/` data or `questions.jsonl`. |
