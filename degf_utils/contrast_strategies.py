"""Contrast strategies for DeGF decoding.

Extracted verbatim from the if/elif chain inside degf_sample.sample() and
greedy_search(). Each strategy owns one way of combining the model's logits
over the real image with its logits over a reference view.

WHY THIS IS SAFE TO EXTRACT
    The arithmetic here is pure tensor computation over two logit tensors. It
    needs no model, no weights and no GPU, so it can be tested directly —
    unlike the decoding loop that calls it, which needs all three.

    Every expression below is copied character-for-character from the original
    inline code. BenchyBench/tests/test_contrast_strategies.py asserts each
    strategy reproduces the original expression bitwise on random tensors, so
    the extraction is verified rather than assumed.

CAUTION — this file determines published numbers. The expressions are the
ICLR 2025 method. Restructuring around them is safe because it is tested;
changing the arithmetic inside them is not routine refactoring. See
BenchyBench/PIPELINES.md before altering anything here.

THE FOUR STRATEGIES
    RitualContrast     additive toward a positive reference view
    VCDContrast        contrastive against a noised image
    M3IDContrast       contrastive with a decay schedule over position
    DiffusionContrast  DeGF proper — switches direction per token on the
                       Jensen-Shannon divergence between the two streams

Only DiffusionContrast carries state (the JS log and branch counters), which
is why the strategies are objects rather than free functions: the decoding
loop needs those counts afterwards for the run log.
"""

import torch
from torch import nn


class ContrastStrategy:
    """Base class: combine two logit streams into corrected logits.

    Subclasses implement combine(). The base also carries the adaptive
    plausibility constraint, which is shared by every mode.
    """

    def combine(self, logits, logits_ref):
        """Return corrected logits.

        Args:
            logits: next-token logits from the real image.
            logits_ref: next-token logits from the reference view.
        """
        raise NotImplementedError

    @staticmethod
    def plausibility_cutoff(logits, beta):
        """Threshold below which a token is excluded.

        Computed from the ORIGINAL logits, never the corrected ones. The
        constraint exists to capture what the model found plausible BEFORE any
        correction, so the correction cannot promote a token the model never
        seriously considered. Applying it to corrected scores would defeat it.
        """
        return torch.log(torch.tensor(beta)) + logits.max(dim=-1, keepdim=True).values

    @staticmethod
    def apply_cutoff(corrected, logits, cutoff):
        """Mask tokens whose ORIGINAL logit falls below `cutoff`."""
        return corrected.masked_fill(logits < cutoff, -float("inf"))


class RitualContrast(ContrastStrategy):
    """Additive: amplify agreement with a positive reference view."""

    def __init__(self, alpha_pos):
        self.alpha_pos = alpha_pos

    def combine(self, logits, logits_ref):
        return (logits + self.alpha_pos * logits_ref)


class VCDContrast(ContrastStrategy):
    """Visual contrastive decoding: push away from a distorted view.

    The weights sum to 1, so the overall scale of the logits is roughly
    preserved rather than inflated by the subtraction.
    """

    def __init__(self, alpha_neg):
        self.alpha_neg = alpha_neg

    def combine(self, logits, logits_ref):
        return (1 + self.alpha_neg) * logits - self.alpha_neg * logits_ref


class M3IDContrast(ContrastStrategy):
    """Contrastive with a schedule that STRENGTHENS over token position.

    Note the direction, which is the opposite of what the name "decay"
    suggests. gamma_t = exp(-0.02t) decays, but it appears as the
    (1 - gamma_t) / gamma_t coefficient, which GROWS:

        t=0    coefficient 0       no correction at all
        t=1    coefficient ~0.02   nearly none
        t=200  coefficient ~53.6   very large

    The rationale is that a VLM's conditioning on the image fades as the
    generated text lengthens — later tokens are increasingly driven by what
    was already written. The visual correction is amplified to counteract that
    drift, rather than being backed off.

    Stateful — `t` advances on every call, so one instance belongs to one
    generation.
    """

    DECAY = -0.02

    def __init__(self, t=0):
        self.t = t

    def combine(self, logits, logits_ref):
        gamma_t = torch.exp(torch.tensor(self.DECAY * self.t))
        result = logits + (logits - logits_ref) * (1 - gamma_t) / gamma_t
        self.t += 1
        return result


class DiffusionContrast(ContrastStrategy):
    """DeGF proper: switch direction per token on JS divergence.

    The method's contribution. Rather than applying a fixed correction, it
    measures how far apart the two distributions are and changes sign:

        JS <  threshold   the reference agrees, so ADD its evidence
        JS >= threshold   the reference disagrees, so SUBTRACT it

    The threshold is a class constant, not a constructor argument, because it
    is not a tunable — it is part of the method as published. Changing it
    changes what the method is.

    Stateful: js_list, js_count and token_count accumulate across the
    generation and are read afterwards for the run log.
    """

    JS_THRESHOLD = 0.1

    def __init__(self, alpha_pos, alpha_neg):
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.js_list = []
        self.js_count = 0      # times the contrastive branch fired
        self.token_count = 0

    @staticmethod
    def js_divergence(logits, logits_ref):
        """Jensen-Shannon divergence between the two next-token distributions.

        Mean KL of each against their midpoint M. Symmetric, unlike KL alone,
        so neither stream is privileged as the reference.
        """
        M = 0.5 * (nn.functional.softmax(logits, dim=-1) + nn.functional.softmax(logits_ref, dim=-1))
        return 0.5 * nn.functional.kl_div(nn.functional.log_softmax(logits, dim=-1), M, reduction='batchmean') + 0.5 * nn.functional.kl_div(nn.functional.log_softmax(logits_ref, dim=-1), M, reduction='batchmean')

    def combine(self, logits, logits_ref):
        js = self.js_divergence(logits, logits_ref)
        self.js_list.append(format(js.item(), '.4f'))

        if js < self.JS_THRESHOLD:
            self.token_count += 1
            return logits + self.alpha_pos * logits_ref

        self.js_count += 1
        self.token_count += 1
        return (1 + self.alpha_neg) * logits - self.alpha_neg * logits_ref
