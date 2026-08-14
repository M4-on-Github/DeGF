"""Text-to-image generation for the DeGF contrastive reference.

DeGF asks the VLM to describe the photograph, regenerates an image from that
description with Stable Diffusion, then contrasts the model's token
distribution over the original against its distribution over the regenerated
reference. Detail that survives only on the original — not on an image built
purely from the model's own words — is evidence the model saw it rather than
assumed it.

Model: runwayml/stable-diffusion-v1-5 in fp16. The alternatives listed in
StableDiffusionGenerator are kept from the upstream implementation as a record
of what was evaluated; switching among them changes the reference image and
therefore the contrastive signal, so it is not a free choice.
"""
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline


class StableDiffusionGenerator:
    """Generates contrastive reference images from text descriptions.

    Owns the pipeline and the long-prompt encoding that goes with it. The
    pipeline loads on first use rather than at construction, so building this
    object is cheap and a baseline run that never generates pays nothing.

    `loader` exists for testing: it lets a fake pipeline be injected so the
    lazy-loading and prompt-encoding contracts can be verified without
    downloading weights or requiring a GPU.
    """

    MODEL_NAME = "runwayml/stable-diffusion-v1-5"
    DEVICE = "cuda:0"          # jobs request one GPU; it is the only one visible
    INFERENCE_STEPS = 50       # upstream default; fewer gives a noisier reference

    #: Evaluated upstream, retained for provenance. Each produces a different
    #: reference image and therefore different published numbers.
    ALTERNATIVES = (
        "CompVis/stable-diffusion-v1-1",
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-xl-base-0.9",
        "stabilityai/stable-diffusion-xl-base-1.0",
    )

    def __init__(self, model_name=None, device=None, loader=None):
        self.model_name = model_name or self.MODEL_NAME
        self.device = device or self.DEVICE
        self._loader = loader or self._default_loader
        self._pipe = None

    @staticmethod
    def _default_loader(model_name):
        # fp16 roughly halves memory against fp32 and lets Stable Diffusion sit
        # alongside LLaVA-1.5-7B on one 48 GB card — which is what makes
        # single-GPU DeGF inference possible at all.
        return StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)

    @property
    def pipe(self):
        """The pipeline, loaded and moved to device on first access."""
        if self._pipe is None:
            self._pipe = self._loader(self.model_name).to(self.device)
        return self._pipe

    @property
    def is_loaded(self):
        return self._pipe is not None

    def encode_prompt(self, prompt, negative_prompt="", device="cuda"):
        """Encode prompts longer than CLIP's 77-token limit.

        Splits the token sequence into chunks, encodes each, and concatenates
        along the sequence axis. Without this the tokenizer truncates, and the
        tail of a long description is silently discarded — dropping exactly
        the trailing detail the contrast exists to test.

        The prompt and negative prompt must yield equal-length embeddings for
        the pipeline to combine them, so whichever is shorter is padded to
        match the longer.

        A fixed chunk boundary can split a phrase across two encoder calls, so
        each chunk is encoded without its neighbours' context. That is the
        upstream approach; the alternative is truncation, which loses the
        content entirely.
        """
        pipeline = self.pipe
        max_length = pipeline.tokenizer.model_max_length

        # simple way to determine length of tokens
        count_prompt = len(prompt.split(" "))
        count_negative_prompt = len(negative_prompt.split(" "))

        # create the tensor based on which prompt is longer
        if count_prompt >= count_negative_prompt:
            input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
            shape_max_length = input_ids.shape[-1]
            negative_ids = pipeline.tokenizer(negative_prompt, truncation=False, padding="max_length",
                                              max_length=shape_max_length, return_tensors="pt").input_ids.to(device)

        else:
            negative_ids = pipeline.tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids.to(device)
            shape_max_length = negative_ids.shape[-1]
            input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False, padding="max_length",
                                           max_length=shape_max_length).input_ids.to(device)

        concat_embeds = []
        neg_embeds = []
        for i in range(0, shape_max_length, max_length):
            concat_embeds.append(pipeline.text_encoder(input_ids[:, i: i + max_length])[0])
            neg_embeds.append(pipeline.text_encoder(negative_ids[:, i: i + max_length])[0])

        return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)

    def generate(self, description):
        """Generate a reference image from a VLM-produced description."""
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(description, "", "cuda")
        return self.pipe(prompt_embeds=prompt_embeds,
                         negative_prompt_embeds=negative_prompt_embeds,
                         num_inference_steps=self.INFERENCE_STEPS).images[0]


# ── Compatibility facade ─────────────────────────────────────────────────────
# run_inference.py and QWEN's degf_ablate call these directly.

def get_image_generation_pipeline():
    """Return a loaded Stable Diffusion pipeline.

    Kept returning the raw pipeline rather than the generator object, because
    callers hold the result and pass it back to
    generate_image_stable_diffusion(). Loading happens here, on the .pipe
    access, matching the original eager behaviour for this entry point.
    """
    return StableDiffusionGenerator().pipe


def generate_image_stable_diffusion(pipe, description):
    """Generate a reference image using an already-loaded `pipe`."""
    generator = StableDiffusionGenerator()
    generator._pipe = pipe          # adopt the caller's pipeline
    return generator.generate(description)


def get_pipeline_embeds(pipeline, prompt, negative_prompt, device):
    """Encode prompts exceeding CLIP's token limit. See encode_prompt()."""
    generator = StableDiffusionGenerator()
    generator._pipe = pipeline
    return generator.encode_prompt(prompt, negative_prompt, device)
