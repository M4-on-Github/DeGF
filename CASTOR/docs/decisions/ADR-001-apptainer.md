# ADR-001: Use Apptainer containers instead of conda for cluster environment

## Status
Accepted

## Date
2026-06-17

## Context
CASTOR runs on the pleiades SLURM cluster (AART Lab, CMU). The DeGF stack has strict
dependency ordering requirements: transformers 4.31.0 and tokenizers 0.13.3 must be
pinned before peft is installed, or peft upgrades transformers to ≥4.35 and breaks
LLaVA's generation monkey-patching. Standard conda or pip installs on a shared cluster
risk these pins being violated by other users' packages in a shared environment.

## Decision
Package the entire runtime in an Apptainer `.sif` container built from `container.def`.
The base image (`pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel`) ships torch pre-installed;
we layer only the additional packages on top with explicit version pins and a post-install
assertion that fails the build if the pins were violated.

## Alternatives Considered

### Conda environment in home directory
- Pros: Familiar, fast iteration
- Cons: Shared cluster may have system packages that shadow pins; activation state
  is fragile across SLURM node boundaries; harder to guarantee reproducibility for
  collaborators or future runs
- Rejected: The strict pin ordering made isolation a hard requirement

### Docker
- Pros: Well-documented, portable
- Cons: Requires root to build; not available on most HPC clusters including pleiades
- Rejected: Apptainer (formerly Singularity) is the HPC-standard rootless alternative

### Module system (`module load`)
- Pros: Zero setup
- Cons: pleiades modules don't provide the exact torch+cuda version combination needed;
  no way to pin transformers/tokenizers through the module system
- Rejected: Insufficient control over package versions

## Consequences
- **Container rebuild guard**: `submit_job.sh` sha256-hashes `container.def` and compares
  it to a stored hash (`$SIF.def.sha256`). The build is skipped when hashes match,
  regardless of file timestamps — safe across `scp`/`rsync` transfers. To force a
  rebuild, update `container.def` or delete both `/data/$USER/castor.sif` and
  `/data/$USER/castor.sif.def.sha256`.
- **SIF stored in `/data/$USER/`**: At ~6 GB the container doesn't fit well in the
  500 GB home quota and is excluded from home backups.
- **Post-install assertion**: `container.def` verifies `transformers==4.31.0`,
  `tokenizers==0.13.x`, and `peft==0.4.x` at build time. A version violation fails
  the build loudly rather than producing a silently broken runtime.
