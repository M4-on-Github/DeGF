"""Stable Diffusion image-variation pipeline.

Produces a visual variant of an input image rather than generating from text.
DeGF uses this as an alternative way to build the contrastive reference: instead
of describing the image and regenerating from that description (see
image_generation.py), it perturbs the image directly, so the reference stays
closer to the original composition.

Model: lambdalabs/sd-image-variations-diffusers, revision v2.0. The revision is
pinned deliberately — v1.0 and v2.0 differ in conditioning and produce visibly
different variants, which would change results.
"""
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms

# Hardcoded rather than derived. The inference jobs request a single GPU, so
# device 0 is the only one visible inside the container.
device = "cuda:0"

def get_image_variation_pipeline():
    """Load the image-variation pipeline onto the GPU.

    Expensive (weights download on a cold cache, then a GPU transfer), so call
    once per process and reuse. run_inference.py holds a single instance for
    the whole run.
    """
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
      "lambdalabs/sd-image-variations-diffusers",
      revision="v2.0",
      )
    sd_pipe = sd_pipe.to(device)
    return sd_pipe

def apply_image_variation(sd_pipe, image, guidance_scale=3):
    """Generate a variation of `image`.

    Args:
        sd_pipe: pipeline from get_image_variation_pipeline().
        image: PIL image. Grayscale input is converted to RGB first — the
            pipeline expects three channels, and some CASTOR photographs are
            genuinely grayscale.
        guidance_scale: how strongly to push toward the conditioning. Higher
            values follow the source image more closely and produce less
            variation; 3 is the upstream default.

    Returns:
        A single PIL image.

    Note:
        The preprocessing is CLIP's, not Stable Diffusion's — 224x224 bicubic
        with CLIP's normalization constants — because this pipeline conditions
        on a CLIP image embedding. Those numbers are not arbitrary and must
        match what the CLIP encoder was trained with; changing them silently
        degrades the conditioning rather than raising.

        antialias=False matches the upstream reference implementation. Enabling
        it changes the embedding slightly and therefore the output.
    """
    # If image is grayscale, convert to RGB
    if image.mode == "L":
        image = image.convert("RGB")
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
          [0.48145466, 0.4578275, 0.40821073],
          [0.26862954, 0.26130258, 0.27577711]),
    ])
    inp = tform(image).to(device).unsqueeze(0)
    out = sd_pipe(inp, guidance_scale=guidance_scale)
    return out["images"][0]
