"""CLIP image-image cosine similarity.

Scores how closely a Stable Diffusion reference image matches the original
photograph. A low score means the generated reference drifted from what the
model was actually looking at, so any contrastive signal derived from it is
less trustworthy.

LAZY LOADING
    CLIP is loaded on FIRST USE, not at import. Previously the processor and
    model were constructed at module scope, so merely importing this file
    downloaded ~600 MB on a cold cache and allocated GPU memory — even though
    nothing in the CASTOR pipeline currently calls it.

    The free function get_clip_similarity() is preserved for existing callers
    and delegates to a module-level CLIPSimilarity instance, so the only
    observable change is WHEN the weights load.
"""
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
import torch.nn as nn


class CLIPSimilarity:
    """Cosine similarity between images in CLIP embedding space.

    The model is loaded on first use and cached on the instance, so repeated
    comparisons in a generation loop pay the load cost once.

    `loader` and `processor_loader` exist for testing: they let a fake model be
    injected so the lazy-loading contract can be verified without downloading
    weights or requiring a GPU.
    """

    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self, model_name=None, device=None,
                 loader=None, processor_loader=None):
        self.model_name = model_name or self.MODEL_NAME
        self._device = device
        self._loader = loader or (lambda name: CLIPModel.from_pretrained(name))
        self._processor_loader = processor_loader or (lambda name: AutoProcessor.from_pretrained(name))
        self._model = None
        self._processor = None

    @property
    def device(self):
        """Resolved on first access, not at construction.

        Deferring matters: constructing this object must not depend on CUDA
        being visible yet, and the container sets device visibility late.
        """
        if self._device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        return self._device

    @property
    def processor(self):
        if self._processor is None:
            self._processor = self._processor_loader(self.model_name)
        return self._processor

    @property
    def model(self):
        if self._model is None:
            self._model = self._loader(self.model_name).to(self.device)
        return self._model

    @property
    def is_loaded(self):
        """True once the weights have actually been materialised."""
        return self._model is not None

    def embed(self, image):
        """Encode one image to a CLIP feature vector."""
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            return self.model.get_image_features(**inputs)

    def similarity(self, image1, image2):
        """Cosine similarity between two images.

        Returns a float in [-1, 1] — in practice [0, 1] for natural images,
        since CLIP embeddings of real photographs rarely oppose each other.

        The images are encoded in separate forward passes rather than as a
        batch. Slower, but peak memory stays flat and no resizing is applied
        to make a batch uniform. Only element 0 of each result is compared.
        """
        image_features1 = self.embed(image1)
        image_features2 = self.embed(image2)
        cos = nn.CosineSimilarity(dim=0)
        return cos(image_features1[0], image_features2[0]).item()


#: Shared instance backing the free function below. Constructing it is cheap —
#: no weights are touched until the first call.
_DEFAULT = CLIPSimilarity()


def get_clip_similarity(image1, image2):
    """Cosine similarity between two images. Facade over CLIPSimilarity."""
    return _DEFAULT.similarity(image1, image2)
