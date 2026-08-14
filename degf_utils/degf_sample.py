"""DeGF decoding — monkey-patched replacements for HuggingFace generation.

Implements "Self-Correcting Decoding with Generative Feedback for Mitigating
Hallucinations in Large Vision-Language Models" (ICLR 2025).

HOW IT IS INSTALLED
    evolve_degf_sampling() overwrites GenerationMixin.sample and
    GenerationMixin.greedy_search on the transformers module itself. The patch
    is global and permanent for the process: after calling it, EVERY model in
    the process decodes through this file, including a baseline run. Baseline
    behaviour is preserved because none of the contrast modes activate unless
    the corresponding flag is set in model_kwargs.

    This is also why transformers is pinned to 4.31.0. These functions are
    copies of that version's originals with the contrast block inserted; a
    different version's generation loop has a different signature and internal
    contract, and the patch would either fail or silently diverge.

THE IDEA
    The model is run twice per token: once on the real image, once on a
    "reference" view of it. Comparing the two next-token distributions
    separates detail grounded in the image from detail the model would assert
    regardless.

FOUR CONTRAST MODES, selected by flags in model_kwargs
    use_ritual     additive toward a positive reference
    use_vcd        contrastive against a noised image (visual contrastive
                   decoding)
    use_m3id       contrastive with a decay schedule, so the correction
                   weakens as the sequence grows
    use_diffusion  DeGF proper — see below

DeGF's JS-DIVERGENCE GATE
    DeGF does not apply a fixed correction. It measures the Jensen-Shannon
    divergence between the two distributions at each token and switches
    direction on it:

      JS < 0.1  the two views agree, so the reference is trustworthy and its
                signal is ADDED (amplifying shared evidence)
      JS >= 0.1 the views disagree, so the reference is treated as a
                distractor and SUBTRACTED (contrastive decoding)

    That per-token switch is the paper's contribution. The threshold is
    hardcoded below rather than exposed as a hyperparameter.

ADAPTIVE PLAUSIBILITY CONSTRAINT
    Before any contrast is applied, tokens whose original logit falls more than
    log(degf_beta) below the maximum are masked to -inf. This stops the
    correction from promoting a token the model considered implausible to begin
    with — without it, subtracting a large reference logit can leave a
    nonsensical token on top.

CAUTION — this file determines published numbers. Its arithmetic is not
routine refactoring material; see BenchyBench/PIPELINES.md for the prerequisite
before changing it.
"""
import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput


def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    # auto-regressive generation
    model_kwargs_pos = model_kwargs.copy()
    model_kwargs_neg = model_kwargs.copy()
    
    print("use_ritual = ", model_kwargs.get("use_ritual"))
    print("use_vcd = ", model_kwargs.get("use_vcd"))
    print("use_m3id = ", model_kwargs.get("use_m3id"))
    print("use_diffusion = ", model_kwargs.get("use_diffusion"))
    
    print("pos = ", model_kwargs.get("degf_alpha_pos"))
    print("neg = ", model_kwargs.get("degf_alpha_neg"))
    print("beta = ", model_kwargs.get("degf_beta"))    
    
    t=0
    js_count = 0
    token_count = 0
    js_list = []
    
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # image_size: 336x336

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions, # True
            output_hidden_states=output_hidden_states
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        
        ## For complementive & contrastive decoding
        use_ritual = model_kwargs.get("use_ritual")
        use_vcd = model_kwargs.get("use_vcd")
        use_m3id = model_kwargs.get("use_m3id")
        use_diffusion = model_kwargs.get("use_diffusion")
        
        if use_ritual or use_vcd or use_m3id or use_diffusion:
            next_token_logits_pos = next_token_logits
            next_token_logits_neg = next_token_logits

            if model_kwargs["images_pos"] is not None and use_ritual:
                model_inputs_pos = self.prepare_inputs_for_generation_pos(input_ids, **model_kwargs_pos)
                outputs_pos = self(
                    **model_inputs_pos,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_pos = outputs_pos.logits[:, -1, :]

            elif model_kwargs["images_neg"] is not None and use_vcd:
                model_inputs_neg = self.prepare_inputs_for_generation_neg(input_ids, **model_kwargs_neg)
                outputs_neg = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
            elif use_m3id:
                model_inputs_neg = self.prepare_inputs_for_generation_m3id(input_ids, **model_kwargs_neg)
                outputs_neg = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
            elif model_kwargs["images_neg"] is not None and use_diffusion:
                model_inputs_neg = self.prepare_inputs_for_generation_neg(input_ids, **model_kwargs_neg)
                outputs_neg = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
                
                
            degf_alpha_pos = model_kwargs.get("degf_alpha_pos") if model_kwargs.get("degf_alpha_pos") is not None else 3
            degf_alpha_neg = model_kwargs.get("degf_alpha_neg") if model_kwargs.get("degf_alpha_neg") is not None else 1
            degf_beta = model_kwargs.get("degf_beta") if model_kwargs.get("degf_beta") is not None else 0.1

            # Adaptive Plausibility Constraint. Tokens sitting more than
            # log(degf_beta) below the top ORIGINAL logit are excluded further
            # down. Computed from next_token_logits (the real image) rather
            # than from the corrected scores on purpose: the constraint is
            # meant to reflect what the model found plausible before any
            # contrast was applied, so the correction cannot promote a token
            # the model never considered.
            cutoff = torch.log(torch.tensor(degf_beta)) + next_token_logits.max(dim=-1, keepdim=True).values

            if use_ritual:
                # Purely additive: amplify agreement with the positive view.
                diffs = (next_token_logits + degf_alpha_pos * next_token_logits_pos)
            elif use_vcd:
                # Visual contrastive decoding: push away from the distorted
                # view. Weights sum to 1 so the scale of the logits is roughly
                # preserved.
                diffs = (1 + degf_alpha_neg) * next_token_logits - degf_alpha_neg * next_token_logits_neg
            elif use_m3id:
                # Correction decays as generation proceeds: later tokens are
                # constrained more by the text produced so far than by the
                # image, so a constant visual correction would over-steer.
                gamma_t = torch.exp(torch.tensor(-0.02*t))
                diffs = next_token_logits + (next_token_logits - next_token_logits_neg)*(1-gamma_t)/gamma_t
                t += 1
            elif use_diffusion:
                # DeGF proper. Jensen-Shannon divergence between the two
                # next-token distributions, computed as the mean KL of each
                # against their midpoint M — symmetric, unlike KL alone, so
                # neither view is privileged.
                M = 0.5 * (nn.functional.softmax(next_token_logits, dim=-1) + nn.functional.softmax(next_token_logits_neg, dim=-1))
                js = 0.5 * nn.functional.kl_div(nn.functional.log_softmax(next_token_logits, dim=-1), M, reduction='batchmean') + 0.5 * nn.functional.kl_div(nn.functional.log_softmax(next_token_logits_neg, dim=-1), M, reduction='batchmean')
                js_list.append(format(js.item(), '.4f'))

                # The gate. Low divergence means the reference agrees, so its
                # evidence is ADDED; high divergence means it disagrees, so it
                # is treated as a distractor and SUBTRACTED. Switching sign
                # per token is what distinguishes DeGF from fixed contrastive
                # decoding.
                #
                # The 0.1 threshold is hardcoded, not a tunable — changing it
                # changes the method, not a setting.
                if js < 0.1: # 0.1
                    token_count += 1
                    diffs = next_token_logits + degf_alpha_pos * next_token_logits_neg
                else:
                    js_count += 1        # counted so the log reports how often
                                         # the contrastive branch fired
                    token_count += 1
                    diffs = (1 + degf_alpha_neg) * next_token_logits - degf_alpha_neg * next_token_logits_neg

            # Apply the plausibility mask. Note the comparison is against the
            # ORIGINAL logits, not `diffs`.
            logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            logits = logits_processor(input_ids, logits)
            logits = logits_warper(input_ids, logits)

            next_token_scores = logits
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        if use_ritual:
            model_kwargs_pos = self._update_model_kwargs_for_generation(
                outputs_pos, model_kwargs_pos, is_encoder_decoder=self.config.is_encoder_decoder
            )
        if use_vcd or use_m3id or use_diffusion:
            model_kwargs_neg = self._update_model_kwargs_for_generation(
                outputs_neg, model_kwargs_neg, is_encoder_decoder=self.config.is_encoder_decoder
            )
            
        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    print("js_count: ", js_count, "| token_count: ", token_count)
    
    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids, js_list
    

def greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    # auto-regressive generation
    model_kwargs_pos = model_kwargs.copy()
    model_kwargs_neg = model_kwargs.copy()
    
    print("use_ritual = ", model_kwargs.get("use_ritual"))
    print("use_vcd = ", model_kwargs.get("use_vcd"))
    print("use_m3id = ", model_kwargs.get("use_m3id"))
    print("use_diffusion = ", model_kwargs.get("use_diffusion"))
    
    print("pos = ", model_kwargs.get("degf_alpha_pos"))
    print("neg = ", model_kwargs.get("degf_alpha_neg"))
    print("beta = ", model_kwargs.get("degf_beta"))    
    
    t=0
    js_count = 0
    token_count = 0
    js_list = []
    
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # image_size: 336x336

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions, # True
            output_hidden_states=output_hidden_states
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        
        ## For complementive & contrastive decoding
        use_ritual = model_kwargs.get("use_ritual")
        use_vcd = model_kwargs.get("use_vcd")
        use_m3id = model_kwargs.get("use_m3id")
        use_diffusion = model_kwargs.get("use_diffusion")
        
        if use_ritual or use_vcd or use_m3id or use_diffusion:
            next_token_logits_pos = next_token_logits
            next_token_logits_neg = next_token_logits

            if model_kwargs["images_pos"] is not None and use_ritual:
                model_inputs_pos = self.prepare_inputs_for_generation_pos(input_ids, **model_kwargs_pos)
                outputs_pos = self(
                    **model_inputs_pos,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_pos = outputs_pos.logits[:, -1, :]

            elif model_kwargs["images_neg"] is not None and use_vcd:
                model_inputs_neg = self.prepare_inputs_for_generation_neg(input_ids, **model_kwargs_neg)
                outputs_neg = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
            elif use_m3id:
                model_inputs_neg = self.prepare_inputs_for_generation_m3id(input_ids, **model_kwargs_neg)
                outputs_neg = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
            elif model_kwargs["images_neg"] is not None and use_diffusion:
                model_inputs_neg = self.prepare_inputs_for_generation_neg(input_ids, **model_kwargs_neg)
                outputs_neg = self(
                    **model_inputs_neg,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states
                )
                next_token_logits_neg = outputs_neg.logits[:, -1, :]
                
                
            degf_alpha_pos = model_kwargs.get("degf_alpha_pos") if model_kwargs.get("degf_alpha_pos") is not None else 3
            degf_alpha_neg = model_kwargs.get("degf_alpha_neg") if model_kwargs.get("degf_alpha_neg") is not None else 1
            degf_beta = model_kwargs.get("degf_beta") if model_kwargs.get("degf_beta") is not None else 0.1

            # set cutoff for Adaptive Plausibility Constraints
            cutoff = torch.log(torch.tensor(degf_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            if use_ritual:
                diffs = (next_token_logits + degf_alpha_pos * next_token_logits_pos)
            elif use_vcd:
                diffs = (1 + degf_alpha_neg) * next_token_logits - degf_alpha_neg * next_token_logits_neg
            elif use_m3id:
                gamma_t = torch.exp(torch.tensor(-0.02*t))
                diffs = next_token_logits + (next_token_logits - next_token_logits_neg)*(1-gamma_t)/gamma_t
                t += 1
            elif use_diffusion:
                # calculate kl divergence
                # kl = nn.functional.kl_div(nn.functional.log_softmax(next_token_logits_neg, dim=-1), nn.functional.softmax(next_token_logits, dim=-1), reduction='batchmean')
                # if kl < 0.5:
                #     diffs = next_token_logits + degf_alpha_pos * next_token_logits_pos
                # else:
                #     diffs = (1 + degf_alpha_neg) * next_token_logits - degf_alpha_neg * next_token_logits_neg
                # diffs = (1 + (kl + 0.5)) * next_token_logits - (kl + 0.5) * next_token_logits_neg
                # calculate js divergence
                M = 0.5 * (nn.functional.softmax(next_token_logits, dim=-1) + nn.functional.softmax(next_token_logits_neg, dim=-1))
                js = 0.5 * nn.functional.kl_div(nn.functional.log_softmax(next_token_logits, dim=-1), M, reduction='batchmean') + 0.5 * nn.functional.kl_div(nn.functional.log_softmax(next_token_logits_neg, dim=-1), M, reduction='batchmean')
                js_list.append(format(js.item(), '.4f'))
                # print("kl:",kl)

                # diffs = (1 + (js + 0.5)) * next_token_logits - (js + 0.5) * next_token_logits_neg
                if js < 0.1: # 0.1
                    token_count += 1
                    diffs = next_token_logits + degf_alpha_pos * next_token_logits_neg
                else:
                    js_count += 1
                    token_count += 1
                    diffs = (1 + degf_alpha_neg) * next_token_logits - degf_alpha_neg * next_token_logits_neg
            
            logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            ## degf_comments: apply temperature warping and top-k filtering in contrastive decoding
            logits = logits_processor(input_ids, logits)
            # logits = logits_warper(input_ids, logits)

            next_token_scores = logits
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            # next_token_scores = logits_warper(input_ids, next_token_scores)
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        if use_ritual:
            model_kwargs_pos = self._update_model_kwargs_for_generation(
                outputs_pos, model_kwargs_pos, is_encoder_decoder=self.config.is_encoder_decoder
            )
        if use_vcd or use_m3id or use_diffusion:
            model_kwargs_neg = self._update_model_kwargs_for_generation(
                outputs_neg, model_kwargs_neg, is_encoder_decoder=self.config.is_encoder_decoder
            )
            
        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    print("js_count: ", js_count, "| token_count: ", token_count)
    
    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids, js_list

def evolve_degf_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample
    transformers.generation.GenerationMixin.greedy_search = greedy_search