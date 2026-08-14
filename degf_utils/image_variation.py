"""Stable Diffusion image-variation pipeline.

Produces a visual variant of an input image rather than generating from text.
DeGF uses this as an alternative way to build the contrastive reference:
instead of describing the image and regenerating from that description (see
image_generation.py), it perturbs the image directly, so the reference stays
closer to the original composition.
"""
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms

# Retained at module scope for callers that referenced it.
device = "cuda:0"


class ImageVariationGenerator:
    """Generates a visual variation of an input image.

    The pipeline loads on first use rather than at construction, so building
    this object is cheap.

    `loader` exists for testing: it lets a fake pipeline be injected so the
    preprocessing and lazy-loading contracts can be verified without weights
    or a GPU.
    """

    MODEL_NAME = "lambdalabs/sd-image-variations-diffusers"
    #: Pinned deliberately — v1.0 and v2.0 differ in conditioning and produce
    #: visibly different variants, which would change results.
    REVISION = "v2.0"
    DEVICE = "cuda:0"
    DEFAULT_GUIDANCE = 3

    #: CLIP's preprocessing, NOT Stable Diffusion's, because this pipeline
    #: conditions on a CLIP image embedding. These constants must match what
    #: the CLIP encoder was trained with; changing them degrades conditioning
    #: silently rather than raising.
    CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
    INPUT_SIZE = (224, 224)

    def __init__(self, model_name=None, revision=None, device=None, loader=None):
        self.model_name = model_name or self.MODEL_NAME
        self.revision = revision or self.REVISION
        self.device = device or self.DEVICE
        self._loader = loader or self._default_loader
        self._pipe = None

    def _default_loader(self, model_name, revision):
        return StableDiffusionImageVariationPipeline.from_pretrained(
            model_name, revision=revision)

    @property
    def pipe(self):
        if self._pipe is None:
            self._pipe = self._loader(self.model_name, self.revision).to(self.device)
        return self._pipe

    @property
    def is_loaded(self):
        return self._pipe is not None

    @classmethod
    def build_transform(cls):
        """CLIP preprocessing pipeline.

        antialias=False matches the upstream reference implementation.
        Enabling it changes the embedding slightly and therefore the output.
        """
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                cls.INPUT_SIZE,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=False,
                ),
            transforms.Normalize(cls.CLIP_MEAN, cls.CLIP_STD),
        ])

    @staticmethod
    def to_rgb(image):
        """Convert grayscale to RGB.

        The pipeline expects three channels, and some CASTOR photographs are
        genuinely grayscale.
        """
        if image.mode == "L":
            return image.convert("RGB")
        return image

    def generate(self, image, guidance_scale=None):
        """Generate a variation of `image`.

        `guidance_scale` controls how strongly to follow the source image:
        higher values stay closer and vary less. 3 is the upstream default.
        """
        guidance_scale = self.DEFAULT_GUIDANCE if guidance_scale is None else guidance_scale
        image = self.to_rgb(image)
        inp = self.build_transform()(image).to(self.device).unsqueeze(0)
        out = self.pipe(inp, guidance_scale=guidance_scale)
        return out["images"][0]


# ── Compatibility facade ─────────────────────────────────────────────────────

def get_image_variation_pipeline():
    """Return a loaded image-variation pipeline."""
    return ImageVariationGenerator().pipe


def apply_image_variation(sd_pipe, image, guidance_scale=3):
    """Generate a variation using an already-loaded `sd_pipe`."""
    generator = ImageVariationGenerator()
    generator._pipe = sd_pipe          # adopt the caller's pipeline
    return generator.generate(image, guidance_scale=guidance_scale)
