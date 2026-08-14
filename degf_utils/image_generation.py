"""Text-to-image generation for the DeGF contrastive reference.

DeGF asks the VLM to describe the photograph, regenerates an image from that
description with Stable Diffusion, then contrasts the model's token
distribution over the original against its distribution over the regenerated
reference. Detail that survives only on the original — not on an image built
purely from the model's own words — is evidence the model saw it rather than
assumed it.

Model: runwayml/stable-diffusion-v1-5 in fp16. The commented alternatives
(SD v1-1, v2-1, SDXL base) are kept from the upstream implementation as a
record of what was evaluated; switching among them changes the reference image
and therefore the contrastive signal, so it is not a free choice.
"""
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline

def get_image_generation_pipeline():
    """Load the Stable Diffusion pipeline onto the GPU.

    Expensive — weights download on a cold cache, then transfer to GPU — so
    call once per process and reuse. fp16 roughly halves memory against fp32
    and fits alongside LLaVA-1.5-7B on a single 48 GB card, which is what makes
    single-GPU DeGF inference possible at all.

    Device is hardcoded to cuda:0: the inference jobs request one GPU, so it is
    the only device visible inside the container.
    """
    # pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-1", torch_dtype=torch.float16)
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    # pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    # pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

    pipe = pipe.to("cuda:0")

    return pipe

def generate_image_stable_diffusion(pipe, description):
    """Generate a reference image from a VLM-produced description.

    Goes through get_pipeline_embeds rather than passing `description`
    straight to the pipeline (see the commented-out line). That matters here:
    VLM descriptions routinely exceed CLIP's 77-token limit, and the direct
    path would silently truncate them — dropping exactly the trailing detail
    the contrast is meant to test.

    50 inference steps is the upstream default. Fewer is faster but yields a
    noisier reference, which weakens the comparison.
    """
    prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(pipe, description, "", "cuda")
    image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_inference_steps=50).images[0]
    # image = pipe(description).images[0]
    return image

def get_pipeline_embeds(pipeline, prompt, negative_prompt, device):
    """ Get pipeline embeds for prompts bigger than the maxlength of the pipe
    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :param device:
    :return:

    Encodes prompts longer than CLIP's 77-token limit by splitting the token
    sequence into chunks, encoding each, and concatenating along the sequence
    axis. Without this the tokenizer truncates and the tail of a long
    description is silently discarded.

    The prompt and negative prompt must produce equal-length embeddings for the
    pipeline to combine them, so the shorter one is padded to match the longer.
    That is what the branch below decides: whichever is longer sets
    shape_max_length, and the other is padded to it.

    Chunking at a fixed boundary can split a phrase across two encoder calls,
    so each chunk is encoded without the context of its neighbours. This is the
    upstream approach and is accepted as-is; the alternative is truncation,
    which loses the content entirely.
    """
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


# model_id = "stabilityai/stable-diffusion-2-1"

# # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
# pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]

# image.save("astronaut_rides_horse.png")
