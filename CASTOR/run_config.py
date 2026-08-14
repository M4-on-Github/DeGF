#!/usr/bin/env python3
"""Configuration loading and CLI overriding for CASTOR inference.

Split out of run_inference.py deliberately. That module imports the vendored
LLaVA stack, which pins transformers==4.31.0 and cannot be imported outside
the container — so anything living inside it is untestable. This file imports
nothing beyond the standard library, so the config layer can be exercised
locally while the model code stays where it is.

The precedence rule is the whole point of this module:

    CLI argument  >  config.json  >  nothing

An argument only overrides when it was explicitly given. argparse defaults are
None precisely so "not supplied" is distinguishable from "supplied as a falsy
value" — without that, `--temperature 0` or `--no-diffusion` would be silently
ignored in favour of the config, which is the kind of bug that produces a
complete, plausible run of the wrong experiment.
"""

import json


class RunConfig:
    """A loaded config.json, with CLI overrides applied on top.

    Wraps the two-section {"paths": {...}, "hyperparameters": {...}} structure
    the CASTOR configs use. Behaves like the underlying dict for existing
    callers, so `cfg["paths"]["model_path"]` still works.
    """

    PATHS = "paths"
    HYPERPARAMETERS = "hyperparameters"

    def __init__(self, data):
        self.data = data
        self.data.setdefault(self.PATHS, {})
        self.data.setdefault(self.HYPERPARAMETERS, {})

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path):
        """Read config.json from `path`."""
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    # ── Sections ─────────────────────────────────────────────────────────────

    @property
    def paths(self):
        return self.data[self.PATHS]

    @property
    def hyperparameters(self):
        return self.data[self.HYPERPARAMETERS]

    # ── Dict compatibility ───────────────────────────────────────────────────

    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    def to_dict(self):
        return self.data

    # ── Overriding ───────────────────────────────────────────────────────────

    def apply_overrides(self, args, path_keys, hp_keys):
        """Apply explicitly-supplied CLI arguments over the loaded config.

        `path_keys` and `hp_keys` are the argparse destination names to
        consider. They are passed in rather than hardcoded because the three
        method repos expose different hyperparameters — DeGF has degf_beta,
        ONLY has ritual_beta — while sharing this merge logic.

        Only values that are not None are applied. An argument absent from
        `args` entirely is skipped, so a repo may pass a key its parser does
        not define without raising.

        Returns self, so the call can be chained.
        """
        for key in path_keys:
            value = getattr(args, key, None)
            if value is not None:
                self.paths[key] = value

        for key in hp_keys:
            value = getattr(args, key, None)
            if value is not None:
                self.hyperparameters[key] = value

        return self

    def resolve(self, section, key):
        """Read a value with environment variables and ~ expanded.

        Config paths contain $USER, which is expanded at runtime rather than
        baked in, so the same config works for any account on the cluster.
        """
        import os
        value = self.data[section][key]
        if isinstance(value, str):
            return os.path.expandvars(os.path.expanduser(value))
        return value

    def __repr__(self):
        return "RunConfig(paths=%d, hyperparameters=%d)" % (
            len(self.paths), len(self.hyperparameters))
