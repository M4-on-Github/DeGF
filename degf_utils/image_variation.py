from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms

device = "cuda:0"

def get_image_variation_pipeline():
    sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
      "lambdalabs/sd-image-variations-diffusers",
      revision="v2.0",
      )
    sd_pipe = sd_pipe.to(device)
    return sd_pipe

def apply_image_variation(sd_pipe, image, guidance_scale=3):
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
