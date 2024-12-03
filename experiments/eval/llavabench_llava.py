import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

# import kornia
from transformers import set_seed
from degf_utils.degf_sample import evolve_degf_sampling
from degf_utils.vcd_add_noise import add_diffusion_noise
from degf_utils.image_variation import get_image_variation_pipeline, apply_image_variation
from degf_utils.image_similarity import get_clip_similarity
from degf_utils.image_generation import get_image_generation_pipeline, generate_image_stable_diffusion
evolve_degf_sampling()

from torchvision.transforms import v2
import random 

import warnings
warnings.filterwarnings(action='ignore')

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if args.use_diffusion:
        pipe = get_image_generation_pipeline()
    for line in tqdm(questions):
    # for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        
        aug_dict = {
            'horizontal flip':v2.RandomHorizontalFlip(p=1),
            'vertical flip':v2.RandomVerticalFlip(p=1),
            'rotation':v2.RandomRotation(degrees=180),
            'color jitter':v2.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5),
            'gaussian blur':v2.GaussianBlur(kernel_size=13, sigma=(1.5, 2.0)),
            'crop':v2.RandomResizedCrop(size=336),
        }
        # For statistics
        pos_aug_counter = {k:0 for k in aug_dict}
        pos_aug_counter.update({None: 0})
    
        image_pos = None
        image_neg = None
        
        if args.use_ritual:
            # ==============================================
            #              Image Transforms
            # ==============================================
            pos_aug = random.choice(list(aug_dict.keys()))
            if pos_aug is not None:
                raw_image_pos = aug_dict[pos_aug](image)
                image_pos = image_processor.preprocess(raw_image_pos, return_tensor='pt')['pixel_values'][0] 
                image_pos = torch.tensor(image_pos)
        
            pos_aug_counter[pos_aug] += 1
            print(f"RITUAL Transformation: {pos_aug}")
            
               
        elif args.use_vcd:
            image_neg = add_diffusion_noise(image_tensor, args.noise_step)
        elif args.use_diffusion:
            conv_out = conv_templates[args.conv_mode].copy()
            qs_original = line["text"].strip().split("\n")[0]
            qs_desc = qs_original #+ " Describe relevant details about this."
            # qs_desc = "Provide a detailed description of the image, covering all visible elements and their interactions, so as to thoroughly answer any potential questions about the image."
            qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs_desc # for opera? setting
            # qu_out = DEFAULT_IMAGE_TOKEN + '\n' + qs + " Please answer this question with one word." # for VCD setting
            conv_out.append_message(conv_out.roles[0], qu_out)
            conv_out.append_message(conv_out.roles[1], None)
            prompt_out = conv_out.get_prompt()
            
            input_ids = tokenizer_image_token(prompt_out, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv_out.sep if conv_out.sep_style != SeparatorStyle.TWO else conv_out.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            # ==============================================
            #                ritual method
            # ==============================================
            with torch.inference_mode():
                with torch.no_grad():
                    output_ids, _ = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        images_pos=(image_pos.unsqueeze(0).half().cuda() if image_pos is not None else None),
                        images_neg=(image_neg.unsqueeze(0).half().cuda() if image_neg is not None else None),
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_new_tokens=1024,
                        use_cache=False,
                        use_ritual=False,
                        use_vcd=False,
                        use_m3id=False,
                        use_diffusion=False,
                        degf_alpha_pos=args.degf_alpha_pos,
                        degf_alpha_neg=args.degf_alpha_neg,
                        degf_beta=args.degf_beta,
                    )
                    
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            description = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            description = description.strip()
            if description.endswith(stop_str):
                description = description[:-len(stop_str)]
            # raw_image = Image.open(image_path[0])
            # raw_image.save('raw_image.png')
            print(f"Question: {qs_desc}")
            print(f"Description: {description}")
            raw_image_neg = generate_image_stable_diffusion(pipe, description)
            raw_image_neg.save('image_neg.png')
            image_neg = image_processor.preprocess(raw_image_neg, return_tensor='pt')['pixel_values'][0]
            image_neg = torch.tensor(image_neg)
        
        conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            with torch.no_grad():
                output_ids, _ = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    images_pos=(image_pos.unsqueeze(0).half().cuda() if image_pos is not None else None),
                    images_neg=(image_neg.unsqueeze(0).half().cuda() if image_neg is not None else None),
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    # max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    use_ritual=args.use_ritual,
                    use_vcd=args.use_vcd,
                    use_m3id=args.use_m3id,
                    use_diffusion=args.use_diffusion,
                    degf_alpha_pos=args.degf_alpha_pos,
                    degf_alpha_neg=args.degf_alpha_neg,
                    degf_beta=args.degf_beta,
                )
                
                
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        print(f"Question: {qs}")
        print(f"Answer: {outputs}")

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_ritual", type=str2bool, default=False)

    parser.add_argument("--use_vcd", type=str2bool, default=False)
    parser.add_argument("--noise_step", type=int, default=500)
    
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    
    parser.add_argument("--use_diffusion", type=str2bool, default=False)
    
    parser.add_argument("--degf_alpha_pos", type=float, default=3)
    parser.add_argument("--degf_alpha_neg", type=float, default=1)
    parser.add_argument("--degf_beta", type=float, default=0.1)

    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
