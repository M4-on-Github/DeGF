#!/usr/bin/env python3
"""
CASTOR inference for maritime disaster images.

Loads LLaVA-1.5 once and processes all images in-process.
Resumable: re-running skips already-written results.

Usage (from repo root):
    python CASTOR/run_inference.py
    python CASTOR/run_inference.py --use-diffusion
    python CASTOR/run_inference.py --use-diffusion --degf-alpha-pos 5.0 --temperature 0.7
    python CASTOR/run_inference.py --config path/to/other.json --answers-file results/run2.jsonl

All CLI flags override their config.json counterpart.
Run `python CASTOR/run_inference.py --help` for the full list.
"""
import argparse
import gc
import json
import os
import re
import sys
import time
import warnings
warnings.filterwarnings("ignore")

# ── Torch compatibility patches ───────────────────────────────────────────────
# Must run before any torch/transformers/diffusers import.
import torch

try:
    torch.library.define("torchvision::nms",
        "(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    torch.library.define("torchvision::roi_align",
        "(Tensor input, Tensor rois, float spatial_scale, int pooled_height, "
        "int pooled_width, int sampling_ratio, bool aligned) -> Tensor")
except Exception:
    pass

for _attr in ("xpu", "mps"):
    if not hasattr(torch, _attr):
        setattr(torch, _attr, type(f"Mock{_attr.upper()}", (),
            {"__getattr__": lambda self, n: (lambda *a, **kw: None)})())

for _dtype in ("float8_e4m3fn", "float8_e5m2"):
    if not hasattr(torch, _dtype):
        setattr(torch, _dtype, type("MockDtype", (), {})())

if not hasattr(torch, "compiler"):
    torch.compiler = type("MockCompiler", (), {
        "disable": lambda self, *a, **kw: (lambda f: f) if not a else a[0]
    })()

try:
    from types import ModuleType
    import torch.distributed
    if not hasattr(torch.distributed, "device_mesh"):
        _dm = ModuleType("torch.distributed.device_mesh")
        sys.modules["torch.distributed.device_mesh"] = _dm
        torch.distributed.device_mesh = _dm
        _dm.DeviceMesh = type("DeviceMesh", (), {})
    try:
        import torch.distributed._functional_collectives as _fc
        if not hasattr(_fc, "AsyncCollectiveTensor"):
            _fc.AsyncCollectiveTensor = type("AsyncCollectiveTensor", (), {})
    except ImportError:
        _fc = ModuleType("torch.distributed._functional_collectives")
        sys.modules["torch.distributed._functional_collectives"] = _fc
        _fc.AsyncCollectiveTensor = type("AsyncCollectiveTensor", (), {})
except ImportError:
    pass

try:
    import transformers
    for _name in ("Cache", "DynamicCache", "EncoderDecoderCache",
                  "Dinov2WithRegistersConfig", "Dinov2WithRegistersModel",
                  "SiglipVisionConfig", "SiglipVisionModel", "SiglipImageProcessor",
                  "ViTMAEConfig", "ViTMAEModel"):
        if not hasattr(transformers, _name):
            setattr(transformers, _name, type(_name, (), {}))
except Exception:
    pass

# ── Device selection ─────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    print("ERROR: CUDA not available — this job requires a GPU. Exiting.", flush=True)
    sys.exit(1)
_DEVICE = "cuda"
_gpu = torch.cuda.get_device_properties(0)
print(f"Device     : {_gpu.name}  ({_gpu.total_memory // 1024**2} MiB)", flush=True)

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)                              # run_config.py
sys.path.insert(0, _REPO)                              # degf_utils/
sys.path.insert(0, os.path.join(_REPO, "experiments")) # llava/

# Config handling lives in its own module so it can be tested without the
# vendored LLaVA stack, which pins transformers==4.31.0 and only imports
# inside the container.
from run_config import RunConfig

# ── Project imports ───────────────────────────────────────────────────────────
from PIL import Image
from tqdm import tqdm
from transformers import set_seed

from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
                              DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from degf_utils.degf_sample import evolve_degf_sampling
from degf_utils.image_generation import (get_image_generation_pipeline,
                                          generate_image_stable_diffusion)
evolve_degf_sampling()


# ─────────────────────────────────────────────────────────────────────────────

def _build_sd_prompt(model_output: str) -> str:
    """
    Try to parse the model's JSON output and distill it into a natural-language
    SD prompt. Falls back to using raw text if parsing fails.
    """
    fields = {}
    try:
        parsed = json.loads(model_output)
        if isinstance(parsed, dict):
            fields = parsed
    except (json.JSONDecodeError, ValueError):
        for key in ("description", "state", "vessel_type",
                    "size_estimate", "cargo", "surroundings"):
            m = re.search(rf'"{key}"\s*:\s*"([^"]*)"', model_output)
            if m:
                fields[key] = m.group(1)

    if not fields:
        return model_output

    desc = fields.get("description", "").strip()
    parts = [desc] if desc else [model_output]

    vessel_type = fields.get("vessel_type", "").strip()
    state       = fields.get("state", "").strip()
    size        = fields.get("size_estimate", "").strip()
    if vessel_type or state or size:
        s = "The vessel is"
        if vessel_type: s += f" a {vessel_type}"
        if state:       s += f" in a {state} state"
        if size:        s += f", estimated {size} in size"
        parts.append(s + ".")

    cargo = fields.get("cargo", "").strip()
    if cargo and cargo.lower() not in ("null", "none", "n/a", ""):
        parts.append(f"Cargo includes {cargo}.")

    surroundings = fields.get("surroundings", "").strip()
    if surroundings:
        parts.append(surroundings)

    return " ".join(parts)


def _run_generate(model, input_ids, image_tensor, image_neg,
                  hp: dict, use_diffusion: bool, use_cache: bool = True):
    with torch.inference_mode(), torch.no_grad():
        output_ids, _ = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().to(_DEVICE),
            images_pos=None,
            images_neg=(image_neg.unsqueeze(0).half().to(_DEVICE)
                        if image_neg is not None else None),
            do_sample=True,
            temperature=hp["temperature"],
            top_p=hp["top_p"],
            top_k=hp["top_k"],
            max_new_tokens=hp["max_new_tokens"],
            use_cache=use_cache,
            use_ritual=False,
            use_vcd=False,
            use_m3id=False,
            use_diffusion=use_diffusion,
            degf_alpha_pos=hp["degf_alpha_pos"],
            degf_alpha_neg=hp["degf_alpha_neg"],
            degf_beta=hp["degf_beta"],
        )
    return output_ids


def _run_generate_safe(model, input_ids, image_tensor, image_neg,
                       hp: dict, use_diffusion: bool, use_cache: bool = True):
    """Calls _run_generate with one OOM-recovery retry before giving up."""
    try:
        return _run_generate(model, input_ids, image_tensor, image_neg,
                             hp, use_diffusion, use_cache)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        tqdm.write("[OOM] Clearing GPU cache and retrying once...")
        gc.collect()
        torch.cuda.empty_cache()
        return _run_generate(model, input_ids, image_tensor, image_neg,
                             hp, use_diffusion, use_cache)


def run(cfg: dict):
    paths = cfg["paths"]
    hp    = cfg["hyperparameters"]

    set_seed(hp["seed"])
    disable_torch_init()

    use_diffusion = hp["use_diffusion"]
    sd_dir = paths.get("sd_dir")
    if sd_dir:
        os.makedirs(sd_dir, exist_ok=True)
    job_start = time.perf_counter()
    print(f"Mode       : {'DeGF (diffusion on)' if use_diffusion else 'Baseline (no diffusion)'}")
    print(f"Model      : {paths['model_path']}")
    print(f"Questions  : {paths['question_file']}")
    print(f"Output     : {paths['answers_file']}")

    t0 = time.perf_counter()
    def _p(key):
        return os.path.expandvars(os.path.expanduser(paths[key]))

    tokenizer, model, image_processor, _ = load_pretrained_model(
        _p("model_path"),
        paths.get("model_base"),
        get_model_name_from_path(_p("model_path")),
    )
    print(f"[timing] Model load     : {time.perf_counter() - t0:.1f}s")

    sd_pipe = None
    if use_diffusion:
        t0 = time.perf_counter()
        print("Loading Stable Diffusion pipeline...")
        sd_pipe = get_image_generation_pipeline()
        print(f"[timing] SD pipe load   : {time.perf_counter() - t0:.1f}s")

    with open(_p("question_file"), encoding="utf-8") as f:
        questions = [json.loads(line) for line in f]

    answers_file = _p("answers_file")
    os.makedirs(os.path.dirname(os.path.abspath(answers_file)), exist_ok=True)

    start_idx = 0
    if os.path.exists(answers_file):
        with open(answers_file, encoding="utf-8") as f:
            start_idx = sum(1 for _ in f)

    if start_idx >= len(questions):
        print(f"All {len(questions)} images already processed.")
        return

    print(f"Processing : {len(questions)} images (resuming from {start_idx})")
    conv_mode    = hp["conv_mode"]
    image_folder = _p("image_folder")
    model_name   = get_model_name_from_path(_p("model_path"))

    question_times = []

    firstpass_file = paths.get("firstpass_file")
    fp_f = None
    if firstpass_file and use_diffusion:
        os.makedirs(os.path.dirname(os.path.abspath(firstpass_file)), exist_ok=True)
        fp_f = open(firstpass_file, "a", encoding="utf-8")

    with open(answers_file, "a", encoding="utf-8") as out_f:
        for item in tqdm(questions[start_idx:], initial=start_idx, total=len(questions)):
            # Initialise all tensor refs to None so the finally block can
            # safely delete them even if an exception fires mid-image.
            raw_image = image_tensor = image_neg = None
            ids_d = out_d = neg_pixels = None
            input_ids = output_ids = None
            q_start = time.perf_counter()
            t_desc = t_sd = t_main = 0.0

            try:
                raw_image = Image.open(
                    os.path.join(image_folder, item["image"])
                ).convert("RGB")
                image_tensor = image_processor.preprocess(
                    raw_image, return_tensors="pt"
                )["pixel_values"][0]

                # ── DeGF: generate reference image via SD ─────────────────────
                if use_diffusion:
                    t0 = time.perf_counter()
                    conv_d = conv_templates[conv_mode].copy()
                    conv_d.append_message(conv_d.roles[0],
                                          DEFAULT_IMAGE_TOKEN + "\n" + hp["sd_desc_prompt"])
                    conv_d.append_message(conv_d.roles[1], None)
                    stop_d = conv_d.sep if conv_d.sep_style != SeparatorStyle.TWO else conv_d.sep2

                    ids_d = tokenizer_image_token(
                        conv_d.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                    ).unsqueeze(0).cuda()
                    out_d = _run_generate_safe(model, ids_d, image_tensor, None,
                                               hp, use_diffusion=False, use_cache=False)
                    desc = tokenizer.batch_decode(
                        out_d[:, ids_d.shape[1]:], skip_special_tokens=True
                    )[0].strip().rstrip(stop_d)
                    t_desc = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    raw_neg  = generate_image_stable_diffusion(sd_pipe, _build_sd_prompt(desc))
                    if sd_dir:
                        img_stem = os.path.splitext(item["image"].replace("/", "_").replace("\\", "_"))[0]
                        raw_neg.save(os.path.join(sd_dir, img_stem + ".png"))
                    neg_pixels = image_processor.preprocess(
                        raw_neg, return_tensors="pt"
                    )["pixel_values"][0]
                    image_neg = (neg_pixels if isinstance(neg_pixels, torch.Tensor)
                                 else torch.tensor(neg_pixels))
                    t_sd = time.perf_counter() - t0

                # ── Main inference pass ───────────────────────────────────────
                question_text = item["text"]
                qs = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n"
                      if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN + "\n"
                      ) + question_text

                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

                input_ids = tokenizer_image_token(
                    conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).cuda()

                # ── Pre-DeGF first pass (baseline, no SD) ────────────────────
                if fp_f and use_diffusion:
                    fp_out = _run_generate_safe(model, input_ids, image_tensor, None,
                                                hp, use_diffusion=False, use_cache=True)
                    fp_answer = tokenizer.batch_decode(
                        fp_out[:, input_ids.shape[1]:], skip_special_tokens=True
                    )[0].strip().rstrip(stop_str).strip()
                    fp_f.write(json.dumps({
                        "question_id": item["question_id"],
                        "image":       item["image"],
                        "prompt":      question_text,
                        "text":        fp_answer,
                        "model_id":    model_name,
                        "method":      "degf_firstpass",
                    }) + "\n")
                    fp_f.flush()

                t0 = time.perf_counter()
                output_ids = _run_generate_safe(model, input_ids, image_tensor, image_neg,
                                                hp, use_diffusion=use_diffusion, use_cache=True)

                answer = tokenizer.batch_decode(
                    output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
                )[0].strip().rstrip(stop_str).strip()
                t_main = time.perf_counter() - t0

                q_total = time.perf_counter() - q_start
                question_times.append(q_total)
                done = len(question_times)
                avg  = sum(question_times) / done
                eta  = (len(questions) - start_idx - done) * avg
                tqdm.write(
                    f"[{item['image']}] "
                    + (f"desc={t_desc:.1f}s  sd={t_sd:.1f}s  " if use_diffusion else "")
                    + f"infer={t_main:.1f}s  total={q_total:.1f}s  "
                    + f"avg={avg:.1f}s  eta={eta/60:.1f}min"
                )

                out_f.write(json.dumps({
                    "question_id" : item["question_id"],
                    "image"       : item["image"],
                    "prompt"      : question_text,
                    "text"        : answer,
                    "model_id"    : model_name,
                    "use_diffusion": use_diffusion,
                    "timing": {
                        "desc_s"  : round(t_desc, 3),
                        "sd_s"    : round(t_sd,   3),
                        "infer_s" : round(t_main, 3),
                        "total_s" : round(time.perf_counter() - q_start, 3),
                    },
                }) + "\n")
                out_f.flush()

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                is_oom = "out of memory" in str(e).lower()
                if is_oom:
                    tqdm.write(f"[OOM-SKIP] {item['image']}: skipped (OOM even after retry)")
                    error_tag = "oom-skip"
                else:
                    tqdm.write(f"[ERROR] {item['image']}: {e}")
                    error_tag = f"error: {e}"
                # Write a placeholder so line-count resume stays aligned with question index.
                out_f.write(json.dumps({
                    "question_id": item["question_id"],
                    "image":       item["image"],
                    "error":       error_tag,
                }) + "\n")
                out_f.flush()

            finally:
                # Always free GPU memory after every image regardless of success/failure.
                # This is the in-process replacement for subprocess isolation.
                del raw_image, image_tensor, image_neg
                del ids_d, out_d, neg_pixels
                del input_ids, output_ids
                gc.collect()
                torch.cuda.empty_cache()

    if fp_f:
        fp_f.close()

    job_elapsed = time.perf_counter() - job_start
    n = len(question_times)
    print(f"\n{'='*50}")
    print(f"  Images processed : {n}")
    print(f"  Total job time   : {job_elapsed/60:.1f} min  ({job_elapsed:.0f}s)")
    if n:
        print(f"  Mean per image   : {sum(question_times)/n:.1f}s")
        print(f"  Min / Max        : {min(question_times):.1f}s / {max(question_times):.1f}s")
        print(f"  Throughput       : {3600/( sum(question_times)/n ):.0f} images/hr")
    print(f"  Results file     : {answers_file}")
    print(f"{'='*50}")


# ─────────────────────────────────────────────────────────────────────────────

# Argparse destinations that may override config.json. Declared as data rather
# than as a wall of near-identical if-statements, so adding a flag means adding
# a name here instead of another line that is easy to mistype or omit.
# The merge itself lives in run_config.py, which imports nothing beyond the
# standard library and is therefore testable outside the container.
_CFG_PATH_KEYS = (
    "model_path", "model_base", "image_folder", "question_file",
    "answers_file", "sd_dir", "firstpass_file",
)
_CFG_HP_KEYS = (
    "conv_mode", "use_diffusion", "sd_desc_prompt",
    "degf_alpha_pos", "degf_alpha_neg", "degf_beta",
    "temperature", "top_p", "top_k", "seed", "max_new_tokens",
)


def _load_config(path: str) -> dict:
    """Load config.json. Facade over RunConfig.load()."""
    return RunConfig.load(path).to_dict()


def _merge(cfg: dict, args) -> dict:
    """Apply any explicitly-set CLI args over the config.

    Only arguments that are not None are applied, so argparse defaults never
    silently beat config.json. See run_config.RunConfig.apply_overrides().
    """
    return RunConfig(cfg).apply_overrides(args, _CFG_PATH_KEYS, _CFG_HP_KEYS).to_dict()


def main():
    parser = argparse.ArgumentParser(
        description="CASTOR: LLaVA-1.5 inference on maritime disaster images, with/without DeGF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Config file ───────────────────────────────────────────────────────────
    parser.add_argument("--config", default=os.path.join(_HERE, "config.json"),
                        metavar="PATH", help="Base config.json; CLI args override it.")
    parser.add_argument("--run-name", default=None, metavar="TAG",
                        help="Label appended to the auto-generated output filename "
                             "(e.g. 'ap5_b02' → answers_llava_degf_ap5_b02.jsonl). "
                             "Ignored when --answers-file is set explicitly.")
    parser.add_argument("--model-tag", default="llava", metavar="TAG",
                        help="Model identifier prefix in the output filename "
                             "(e.g. 'qwen3vl8b' → answers_qwen3vl8b_degf.jsonl). "
                             "Ignored when --answers-file is set explicitly.")

    # ── Paths ─────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("paths (override config.json)")
    g.add_argument("--model-path",    default=None, metavar="PATH")
    g.add_argument("--model-base",    default=None, metavar="PATH",
                   help="Base model path for LoRA/adapter; null = none.")
    g.add_argument("--image-folder",  default=None, metavar="PATH")
    g.add_argument("--question-file", default=None, metavar="PATH")
    g.add_argument("--answers-file",  default=None, metavar="PATH")
    g.add_argument("--sd-dir",        default=None, metavar="PATH",
                   help="Directory to save SD reference images (DeGF only). "
                        "Skipped when omitted or when running baseline.")
    g.add_argument("--firstpass-file", default=None, metavar="PATH",
                   help="JSONL to save pre-DeGF baseline answers (DeGF only).")

    # ── Mode ──────────────────────────────────────────────────────────────────
    g = parser.add_argument_group("mode")
    mx = g.add_mutually_exclusive_group()
    mx.add_argument("--use-diffusion", dest="use_diffusion", action="store_true",
                    default=None, help="Enable DeGF (overrides config).")
    mx.add_argument("--no-diffusion",  dest="use_diffusion", action="store_false",
                    help="Disable DeGF / baseline mode (overrides config).")

    # ── DeGF hyperparameters ──────────────────────────────────────────────────
    g = parser.add_argument_group("DeGF hyperparameters (override config.json)")
    g.add_argument("--sd-desc-prompt", default=None, metavar="TEXT",
                   help="Prompt used for the SD description pass.")
    g.add_argument("--degf-alpha-pos", type=float, default=None, metavar="F",
                   help="α+ weight: complementary decoding (JS < 0.1).")
    g.add_argument("--degf-alpha-neg", type=float, default=None, metavar="F",
                   help="α- weight: contrastive decoding (JS ≥ 0.1).")
    g.add_argument("--degf-beta",      type=float, default=None, metavar="F",
                   help="Plausibility filter cutoff.")

    # ── Sampling ──────────────────────────────────────────────────────────────
    g = parser.add_argument_group("sampling (override config.json)")
    g.add_argument("--conv-mode",     default=None,
                   help="Conversation template (e.g. llava_v1, llava_llama_2).")
    g.add_argument("--temperature",   type=float, default=None, metavar="F")
    g.add_argument("--top-p",         type=float, default=None, metavar="F")
    g.add_argument("--top-k",         type=int,   default=None, metavar="N")
    g.add_argument("--seed",          type=int,   default=None, metavar="N")
    g.add_argument("--max-new-tokens",type=int,   default=None, metavar="N")

    args = parser.parse_args()

    cfg = _load_config(args.config)
    cfg = _merge(cfg, args)

    # Auto-suffix the output file to prevent runs from clobbering each other.
    # Pattern: answers_{model_tag}_{mode}[_{run_name}].jsonl
    if args.answers_file is None:
        p = cfg["paths"]
        base, ext = os.path.splitext(p["answers_file"])
        mode_tag  = "degf" if cfg["hyperparameters"]["use_diffusion"] else "baseline"
        name_tag  = f"_{args.run_name}" if args.run_name else ""
        p["answers_file"] = f"{base}_{args.model_tag}_{mode_tag}{name_tag}{ext}"

    run(cfg)


if __name__ == "__main__":
    main()
