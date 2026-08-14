"""CLIP image-image cosine similarity.

Used by DeGF to score how closely the Stable Diffusion reference image matches
the original photograph. A low score means the generated reference drifted from
what the model was actually looking at, so the contrastive signal derived from
it is less trustworthy.

WARNING — IMPORT HAS SIDE EFFECTS
    The CLIP processor and model are loaded at module import time, not on first
    use. Importing this module therefore downloads weights on a cold cache and
    allocates GPU memory, even if get_clip_similarity() is never called.

    That is why run_inference.py imports it lazily, inside the diffusion branch,
    rather than at the top of the file: a baseline run must not pay for CLIP.
    Keep it that way when adding callers.

Model: openai/clip-vit-base-patch32 (~600 MB). Device is chosen once at import
and never revisited, so it cannot follow a later device change.
"""
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel
import torch.nn as nn

# Loaded once at import. See the warning above.
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

def get_clip_similarity(image1, image2):
    """Cosine similarity between two images in CLIP embedding space.

    Args:
        image1, image2: PIL images, or anything the CLIP processor accepts.

    Returns:
        float in [-1, 1] — in practice [0, 1] for natural images, since CLIP
        embeddings of real photographs rarely oppose each other. Higher means
        more semantically similar.

    Note:
        The two images are encoded in separate forward passes rather than as a
        batch. Slower, but it keeps peak memory flat and avoids any resizing
        the processor would apply to make a batch uniform.

        Similarity is computed on element 0 of each batch, so only the first
        image of a batched input is compared.
    """
    #Extract features from image1
    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        image_features1 = model.get_image_features(**inputs1)

    #Extract features from image2
    with torch.no_grad():
        inputs2 = processor(images=image2, return_tensors="pt").to(device)
        image_features2 = model.get_image_features(**inputs2)

    #Compute their cosine similarity and convert it into a score between 0 and 1
    cos = nn.CosineSimilarity(dim=0)
    sim = cos(image_features1[0],image_features2[0]).item()

    return sim
