#!/bin/bash
# benchybench_paths.sh — canonical path resolution for CASTOR pipelines
#
# Source from a submit script, then resolve once:
#     source "$REPO/CASTOR/benchybench_paths.sh"
#     BENCHYBENCH_ROOT="$(bb_resolve_root)" || exit 1
#     export BENCHYBENCH_ROOT              # propagate to the batch job
#     IMAGE_DIR="$(bb_images_dir)"
#
# WHY THIS EXISTS
# ---------------
# "Where are the images?" was previously answered five different ways across
# the repos — $(dirname $REPO), $BENCHYBENCH_ROOT, a cwd-relative path in
# config.json, hardcoded /home/$USER paths, and $SLURM_SUBMIT_DIR. Three of
# those can succeed with the WRONG directory, which is worse than failing:
# a pipeline reading a stale image copy produces plausible results nobody
# questions.
#
# The rule here is that every candidate is PROBED before it is accepted, and
# nothing falls through to a guess. If no candidate holds the image set, this
# errors rather than picking something plausible.
#
# ON $0 AND SLURM
# ---------------
# SLURM copies a batch script to a spool directory before running it, so $0
# and BASH_SOURCE inside a running job may point at /var/spool/..., not the
# repo. Script-location derivation is therefore the LAST resort, not the
# first, and $SLURM_SUBMIT_DIR is consulted before it — but validated, which
# is the part that was previously missing.
#
# An identical copy lives in DeGF, ONLY, and QWEN-Maritime. test_paths.sh
# asserts the copies are byte-identical; do not edit one in isolation.

# Path from the BenchyBench root to the canonical image set / ground truth.
BB_IMAGES_SUBPATH="shipwreck_wiki_images/sorted_images"
BB_GT_SUBPATH="Eval_CASTOR/human_ground_truth_label/human_gt.csv"


# True if $1 looks like a BenchyBench root (i.e. actually holds the images).
bb_is_root() {
    [[ -n "$1" && -d "$1/$BB_IMAGES_SUBPATH" ]]
}


# Absolute path of the repo containing this library (<repo>/CASTOR/.. == <repo>).
# Unreliable inside a SLURM batch job; see the note above.
bb_repo_root() {
    cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd
}


# Resolve the BenchyBench root — the directory holding shipwreck_wiki_images/.
#
# Candidates, in order, each accepted only if it actually holds the images:
#   1. $BENCHYBENCH_ROOT          explicit; if set but invalid this is a HARD
#                                 ERROR, never a silent fallback, because a
#                                 wrong explicit value means a wrong assumption
#   2. $SLURM_SUBMIT_DIR          where sbatch was called (job submitted from
#                                 the BenchyBench root)
#   3. dirname($SLURM_SUBMIT_DIR) job submitted from inside a method repo
#   4. parent of this repo        nested / submodule layout
#   5. this repo                  standalone layout, images carried in-repo
#   6. hard error
bb_resolve_root() {
    local repo up

    # 1. Explicit override — trusted enough to fail on, not to guess past.
    if [[ -n "${BENCHYBENCH_ROOT:-}" ]]; then
        if bb_is_root "$BENCHYBENCH_ROOT"; then
            echo "$BENCHYBENCH_ROOT"
            return 0
        fi
        echo "ERROR: BENCHYBENCH_ROOT is set to '$BENCHYBENCH_ROOT'" >&2
        echo "       but '$BENCHYBENCH_ROOT/$BB_IMAGES_SUBPATH' does not exist." >&2
        echo "       Unset it to auto-detect, or point it at the BenchyBench root." >&2
        return 1
    fi

    # 2/3. Submission directory, and one level up. Probed, never assumed —
    #      blindly trusting this is the bug that sent jobs at the wrong tree.
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        if bb_is_root "$SLURM_SUBMIT_DIR"; then
            echo "$SLURM_SUBMIT_DIR"
            return 0
        fi
        up="$(dirname "$SLURM_SUBMIT_DIR")"
        if bb_is_root "$up"; then
            echo "$up"
            return 0
        fi
    fi

    # 4/5. Script location. Last resort: unreliable under SLURM, correct when
    #      the script is executed directly.
    repo="$(bb_repo_root)"
    if [[ -n "$repo" ]]; then
        up="$(dirname "$repo")"
        if bb_is_root "$up"; then          # nested / submodule
            echo "$up"
            return 0
        fi
        if bb_is_root "$repo"; then        # standalone
            echo "$repo"
            return 0
        fi
    fi

    echo "ERROR: cannot locate '$BB_IMAGES_SUBPATH'." >&2
    echo "       Tried, in order:" >&2
    echo "         BENCHYBENCH_ROOT   ${BENCHYBENCH_ROOT:-<unset>}" >&2
    echo "         SLURM_SUBMIT_DIR   ${SLURM_SUBMIT_DIR:-<unset>}" >&2
    echo "         repo parent        ${repo:+$(dirname "$repo")}" >&2
    echo "         repo itself        ${repo:-<undetermined>}" >&2
    echo "       Set BENCHYBENCH_ROOT to the directory containing" >&2
    echo "       shipwreck_wiki_images/, or pass --image-folder explicitly." >&2
    return 1
}


# Absolute path to the image set. Fails if the root cannot be resolved.
bb_images_dir() {
    local root
    root="$(bb_resolve_root)" || return 1
    echo "$root/$BB_IMAGES_SUBPATH"
}


# Absolute path to the ground-truth CSV. Reports a distinct error when the file
# is missing, rather than searching another tree for it.
bb_gt_csv() {
    local root path
    root="$(bb_resolve_root)" || return 1
    path="$root/$BB_GT_SUBPATH"
    if [[ ! -f "$path" ]]; then
        echo "ERROR: ground-truth CSV not found at $path" >&2
        echo "       Is the Eval_CASTOR submodule initialised?" >&2
        echo "       git submodule update --init Eval_CASTOR" >&2
        return 1
    fi
    echo "$path"
}


# Locate this library from a SLURM batch script, where $0 is a spool copy and
# tells us nothing. Echoes the library path, or fails.
#
# Chicken-and-egg: a batch script must find this file before it can source it,
# so this logic is duplicated as a short prologue in each submit_job.sh. Keep
# the two in sync; test_paths.sh exercises the same candidate order.
#
#   $1 repo directory name (DeGF / ONLY / QWEN-Maritime)
bb_locate_lib() {
    local repo_name="$1" cand
    for cand in "$(dirname "${0:-}")/benchybench_paths.sh" \
                "${SLURM_SUBMIT_DIR:-}/CASTOR/benchybench_paths.sh" \
                "${SLURM_SUBMIT_DIR:-}/benchybench_paths.sh" \
                "${BENCHYBENCH_ROOT:-}/$repo_name/CASTOR/benchybench_paths.sh"; do
        if [[ -f "$cand" ]]; then
            echo "$cand"
            return 0
        fi
    done
    return 1
}


# Print the resolved layout. Call this from submit scripts so every job log
# records which tree it actually used — the evidence that is missing when a run
# silently reads a stale copy.
bb_report() {
    local root images
    root="$(bb_resolve_root)" || return 1
    images="$root/$BB_IMAGES_SUBPATH"
    echo "BenchyBench root : $root"
    echo "Images           : $images ($(find "$images" -type f 2>/dev/null | wc -l) files)"
}
